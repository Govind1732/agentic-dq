import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import Joi from 'joi';
import { EventEmitter } from 'events';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Express app
const app = express();
const server = createServer(app);

// Socket.IO server with updated CORS
const io = new Server(server, {
  cors: {
    origin: ["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://localhost:5177", "http://localhost:3000"],
    methods: ["GET", "POST"],
    credentials: true
  },
  // Add connection limits and timeouts
  pingTimeout: 60000,
  pingInterval: 25000,
  maxHttpBufferSize: 1e6, // 1MB max message size
  allowEIO3: true
});

// Middleware
app.use(cors({
  origin: ["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://localhost:5177", "http://localhost:3000"],
  credentials: true,
  optionsSuccessStatus: 200
}));

// Validation schemas
const rcaDataSchema = Joi.object({
  failed_table: Joi.string().min(1).max(255).required(),
  failed_column: Joi.string().min(1).max(255).required(),
  db_type: Joi.string().valid('GCP', 'Teradata').required(),
  validation_query: Joi.string().min(1).max(10000).required(),
  sd_threshold: Joi.alternatives().try(Joi.string().allow(''), Joi.number()),
  expected_std_dev: Joi.alternatives().try(Joi.string().allow(''), Joi.number()),
  expected_value: Joi.alternatives().try(Joi.string().allow(''), Joi.number()),
  actual_value: Joi.alternatives().try(Joi.string().allow(''), Joi.number()),
  execution_date: Joi.alternatives().try(Joi.string().allow(''), Joi.date())
}).unknown(true); // Allow additional fields

const messageSchema = Joi.object({
  message: Joi.alternatives().try(
    Joi.string().min(1).max(50000), // Plain text message
    rcaDataSchema, // RCA JSON object
    Joi.object().unknown(true) // Other structured data
  ).required(),
  timestamp: Joi.date().iso().optional()
}).unknown(true);

// Rate limiting
const rateLimitMap = new Map();
const RATE_LIMIT_WINDOW = 60000; // 1 minute
const RATE_LIMIT_MAX_REQUESTS = 30; // Max 30 requests per minute per IP/socket

// Buffer management
const MESSAGE_BUFFER_SIZE = 1024 * 1024; // 1MB buffer per connection
const connectionBuffers = new Map();

// Process management
class ProcessManager extends EventEmitter {
  constructor() {
    super();
    this.processes = new Map();
    this.processTimeouts = new Map();
    this.maxProcessTime = 300000; // 5 minutes max process time
  }

  addProcess(socketId, process) {
    // Clean up existing process
    this.killProcess(socketId);
    
    this.processes.set(socketId, process);
    
    // Set timeout for process
    const timeout = setTimeout(() => {
      console.log(`Process timeout for socket ${socketId}`);
      this.killProcess(socketId);
      this.emit('processTimeout', socketId);
    }, this.maxProcessTime);
    
    this.processTimeouts.set(socketId, timeout);
  }

  killProcess(socketId) {
    const process = this.processes.get(socketId);
    if (process && !process.killed) {
      try {
        process.kill('SIGTERM');
        setTimeout(() => {
          if (!process.killed) {
            process.kill('SIGKILL');
          }
        }, 5000);
      } catch (error) {
        console.error(`Error killing process for socket ${socketId}:`, error);
      }
    }
    
    this.processes.delete(socketId);
    
    const timeout = this.processTimeouts.get(socketId);
    if (timeout) {
      clearTimeout(timeout);
      this.processTimeouts.delete(socketId);
    }
  }

  getProcess(socketId) {
    return this.processes.get(socketId);
  }

  cleanup() {
    for (const socketId of this.processes.keys()) {
      this.killProcess(socketId);
    }
  }
}

// Validation and sanitization functions
function validateAndSanitizeInput(data, schema) {
  const { error, value } = schema.validate(data, {
    stripUnknown: true,
    abortEarly: false
  });
  
  if (error) {
    throw new Error(`Validation error: ${error.details.map(d => d.message).join(', ')}`);
  }
  
  return value;
}

function checkRateLimit(identifier) {
  const now = Date.now();
  const windowStart = now - RATE_LIMIT_WINDOW;
  
  if (!rateLimitMap.has(identifier)) {
    rateLimitMap.set(identifier, []);
  }
  
  const requests = rateLimitMap.get(identifier);
  
  // Remove old requests outside the window
  const validRequests = requests.filter(timestamp => timestamp > windowStart);
  
  if (validRequests.length >= RATE_LIMIT_MAX_REQUESTS) {
    return false;
  }
  
  validRequests.push(now);
  rateLimitMap.set(identifier, validRequests);
  return true;
}

function sanitizeFilePaths(inputPath) {
  // Prevent directory traversal attacks
  const sanitized = path.normalize(inputPath).replace(/^(\.\.[\/\\])+/, '');
  return sanitized;
}

function createSecureBuffer(socketId) {
  if (!connectionBuffers.has(socketId)) {
    connectionBuffers.set(socketId, {
      data: Buffer.alloc(MESSAGE_BUFFER_SIZE),
      position: 0,
      overflow: false
    });
  }
  return connectionBuffers.get(socketId);
}

// ProcessManager should be declared after the class definition above
const processManager = new ProcessManager();

// Security headers middleware
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  next();
});

// Rate limiting middleware for HTTP requests
app.use((req, res, next) => {
  const clientId = req.ip || req.connection.remoteAddress;
  
  if (!checkRateLimit(`http_${clientId}`)) {
    return res.status(429).json({ 
      error: 'Too many requests',
      message: 'Rate limit exceeded. Please try again later.' 
    });
  }
  
  next();
});

// Express JSON middleware
app.use(express.json({ limit: '10mb' })); // Limit JSON payload size
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log(`Client connected: ${socket.id}`);
  
  // Create secure buffer for this connection
  createSecureBuffer(socket.id);
  
  // Rate limiting for socket connections
  const clientId = socket.handshake.address;
  
  socket.on('disconnect', (reason) => {
    console.log(`Client disconnected: ${socket.id}, reason: ${reason}`);
    
    // Clean up resources
    processManager.killProcess(socket.id);
    connectionBuffers.delete(socket.id);
  });

  // Handle start process request from frontend with validation
  socket.on('start_process', (userData) => {
    try {
      // Rate limiting check
      if (!checkRateLimit(`socket_${socket.id}`)) {
        socket.emit('error', {
          type: 'rate_limit',
          message: 'Too many requests. Please wait before sending another request.'
        });
        return;
      }

      // Validate input data
      const validatedData = validateAndSanitizeInput(userData, messageSchema);
      console.log('Starting RCA process for validated data:', JSON.stringify(validatedData).substring(0, 200) + '...');
      
      startRCAProcess(socket, validatedData);
      
    } catch (error) {
      console.error('Validation error:', error.message);
      socket.emit('error', {
        type: 'validation',
        message: 'Invalid input data: ' + error.message
      });
    }
  });

  // Handle stop process request from frontend
  socket.on('stop_process', () => {
    try {
      console.log(`Stop process requested for socket ${socket.id}`);
      
      // Kill the Python process for this socket
      const process = processManager.getProcess(socket.id);
      if (process) {
        processManager.killProcess(socket.id);
        
        socket.emit('process_step', {
          title: 'Process Stopped',
          type: 'warning',
          content: 'Analysis process has been stopped by user request.',
          timestamp: new Date().toISOString()
        });
        
        socket.emit('final_response', {
          content: 'Process terminated by user.',
          timestamp: new Date().toISOString()
        });
        
        console.log(`Process stopped successfully for socket ${socket.id}`);
      } else {
        socket.emit('process_step', {
          title: 'No Active Process',
          type: 'info',
          content: 'No active process found to stop.',
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      console.error('Error stopping process:', error.message);
      socket.emit('error', {
        type: 'stop_error',
        message: 'Failed to stop process: ' + error.message
      });
    }
  });

  // Handle potential buffer overflow
  socket.on('error', (error) => {
    console.error(`Socket error for ${socket.id}:`, error);
    processManager.killProcess(socket.id);
  });
});

// HTTP POST endpoint for starting RCA with validation
app.post('/api/start-rca', (req, res) => {
  try {
    // Validate input data
    const validatedData = validateAndSanitizeInput(req.body, messageSchema);
    console.log('RCA process started via HTTP with validated data:', JSON.stringify(validatedData).substring(0, 200) + '...');
    
    // For HTTP requests, we'll emit to all connected sockets
    // You could modify this to be more specific if needed
    io.emit('process_started', validatedData);
    
    res.json({ 
      status: 'success',
      message: 'RCA process started', 
      data: validatedData,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('HTTP validation error:', error.message);
    res.status(400).json({
      status: 'error',
      message: 'Invalid input data: ' + error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Function to start the Python RCA process with enhanced security
function startRCAProcess(socket, userData) {
  try {
    // Clean up any existing process for this socket
    processManager.killProcess(socket.id);

    // Sanitize and validate file paths
    const baseDir = sanitizeFilePaths(path.join(__dirname, '..', 'adq-python-script'));
    const scriptsDir = sanitizeFilePaths(path.join(baseDir, 'scripts'));
    const pythonScriptPath = sanitizeFilePaths(path.join(scriptsDir, 'adq_agents.py'));
    const pythonVenvPath = sanitizeFilePaths(path.join(baseDir, '.venv', 'Scripts', 'python.exe'));
    
    console.log('Python script path:', pythonScriptPath);
    console.log('Python venv path:', pythonVenvPath);
    
    // Validate that paths are within expected directories (security check)
    if (!pythonScriptPath.startsWith(scriptsDir) || !pythonVenvPath.startsWith(baseDir)) {
      throw new Error('Invalid file path detected - potential security issue');
    }
    
    // Extract and validate the actual data from userData.message if it exists
    let dataToPass = userData;
    if (userData.message) {
      try {
        // Handle different message types safely
        if (typeof userData.message === 'string') {
          // Try to parse as JSON first
          try {
            const parsedMessage = JSON.parse(userData.message);
            // Validate parsed JSON against RCA schema if it looks like RCA data
            if (parsedMessage.failed_table || parsedMessage.db_type) {
              validateAndSanitizeInput(parsedMessage, rcaDataSchema);
            }
            dataToPass = parsedMessage;
          } catch (jsonError) {
            // If not valid JSON, treat as plain text (already validated by messageSchema)
            dataToPass = { message: userData.message };
          }
        } else if (typeof userData.message === 'object') {
          // Validate object data
          if (userData.message.failed_table || userData.message.db_type) {
            validateAndSanitizeInput(userData.message, rcaDataSchema);
          }
          dataToPass = userData.message;
        }
        
        console.log('Processed message type:', typeof dataToPass);
      } catch (e) {
        console.error('Message processing error:', e.message);
        socket.emit('error', {
          type: 'processing',
          message: 'Error processing message: ' + e.message
        });
        return;
      }
    }
    
    // Convert the correct data to JSON string for Python script argument
    // Limit the size of data being passed
    const userDataJson = JSON.stringify(dataToPass);
    if (userDataJson.length > 50000) { // 50KB limit
      throw new Error('Input data too large. Please reduce the size of your request.');
    }
    
    console.log('Data size being passed to Python:', userDataJson.length, 'characters');
    
    // Configure Python process spawn options with security considerations
    const pythonArgs = [
      '-u', // Unbuffered output
      path.basename(pythonScriptPath),
      userDataJson
    ];
    
    const spawnOptions = {
      cwd: path.dirname(pythonScriptPath),
      env: { 
        ...process.env, 
        PYTHONPATH: scriptsDir,
        PYTHONIOENCODING: 'utf-8',  // Force UTF-8 encoding
        PYTHONUTF8: '1'             // Enable UTF-8 mode in Python
      },
      stdio: ['ignore', 'pipe', 'pipe'] // stdin, stdout, stderr
    };

    // Start the Python process using spawn for better control
    const pythonProcess = spawn(pythonVenvPath, pythonArgs, spawnOptions);
    console.log('Starting Python process with command:', pythonVenvPath, pythonArgs.slice(0, -1), '[data]');
    
    // Store the process reference with enhanced management
    processManager.addProcess(socket.id, pythonProcess);

    // Emit initial status
    socket.emit('process_step', {
      title: 'RCA Process Started',
      type: 'status',
      content: 'Initializing Agentic Data Quality analysis...',
      timestamp: new Date().toISOString()
    });

    // Handle Python script stdout output with delimited protocol parsing
    let messageBuffer = Buffer.alloc(0);
    let expectedLength = null;
    let lengthStr = '';
    let parsingLength = true;

    pythonProcess.stdout.on('data', (data) => {
      try {
        messageBuffer = Buffer.concat([messageBuffer, data]);
        
        while (messageBuffer.length > 0) {
          if (parsingLength) {
            // Look for newline to complete the length
            const newlineIndex = messageBuffer.indexOf('\n');
            if (newlineIndex === -1) {
              // No complete length yet, wait for more data
              break;
            }
            
            lengthStr = messageBuffer.subarray(0, newlineIndex).toString('utf-8');
            expectedLength = parseInt(lengthStr, 10);
            
            if (isNaN(expectedLength) || expectedLength <= 0 || expectedLength > MESSAGE_BUFFER_SIZE) {
              console.error('Invalid message length:', lengthStr);
              // Skip to next newline and try again
              messageBuffer = messageBuffer.subarray(newlineIndex + 1);
              continue;
            }
            
            messageBuffer = messageBuffer.subarray(newlineIndex + 1);
            parsingLength = false;
          }
          
          if (!parsingLength && expectedLength !== null) {
            // We need the message content plus trailing newline
            if (messageBuffer.length < expectedLength + 1) {
              // Not enough data yet, wait for more
              break;
            }
            
            const messageBytes = messageBuffer.subarray(0, expectedLength);
            const message = messageBytes.toString('utf-8');
            
            // Skip the trailing newline
            messageBuffer = messageBuffer.subarray(expectedLength + 1);
            
            // Reset for next message
            parsingLength = true;
            expectedLength = null;
            lengthStr = '';
            
            // Process the complete message
            processDelimitedMessage(socket, message);
          }
        }
      } catch (error) {
        console.error('Error processing Python output:', error);
        socket.emit('process_step', {
          title: 'Processing Error',
          type: 'error',
          content: `Error processing output: ${error.message}`,
          timestamp: new Date().toISOString()
        });
      }
    });

    function processDelimitedMessage(socket, message) {
      try {
        // Check buffer size to prevent memory issues
        const buffer = createSecureBuffer(socket.id);
        const messageSize = Buffer.byteLength(message, 'utf8');
        
        if (buffer.position + messageSize > MESSAGE_BUFFER_SIZE) {
          console.warn(`Buffer overflow for socket ${socket.id}, truncating message`);
          buffer.overflow = true;
          socket.emit('warning', {
            type: 'buffer_overflow',
            message: 'Output buffer full. Some messages may be truncated.'
          });
          return;
        }
        
        // Add to buffer
        buffer.data.write(message, buffer.position, 'utf8');
        buffer.position += messageSize;
        console.log('Python output:', message);
        
        // Try to parse as JSON first (for structured updates)
        let parsedMessage;
        try {
          parsedMessage = JSON.parse(message);
        } catch (e) {
          // If not JSON, treat as plain text
          parsedMessage = {
            title: 'Process Update',
            type: 'display_message',
            content: message
          };
        }

        // Handle different message types based on parsed content
        if (parsedMessage.type === 'NODE_STATUS') {
          // Handle node status updates
          socket.emit('node_status_update', {
            nodeId: parsedMessage.nodeId,
            status: parsedMessage.status,
            message: parsedMessage.message,
            timestamp: new Date().toISOString()
          });
          return;
        }

        if (parsedMessage.type === 'FLOW_UPDATE') {
          // Handle flow updates from emit_function_update
          const flowData = parsedMessage.data;
          socket.emit('process_step', {
            title: flowData.stepId || 'Process Update',
            type: 'flow_update',
            content: flowData.message || 'Processing...',
            data: flowData.data,
            status: flowData.status,
            timestamp: flowData.timestamp || new Date().toISOString()
          });
          return;
        }

        if (parsedMessage.type === 'LINEAGE_TREE') {
          // Handle lineage tree updates
          socket.emit('lineage_tree', {
            nodes: parsedMessage.nodes,
            edges: parsedMessage.edges,
            timestamp: new Date().toISOString()
          });
          return;
        }

        // Check if this is a special flow update (from emit_function_update)
        if (message.startsWith('FLOW_UPDATE:')) {
          const flowData = JSON.parse(message.replace('FLOW_UPDATE:', ''));
          socket.emit('process_step', {
            title: flowData.stepId || 'Process Update',
            type: 'flow_update',
            content: flowData.message || 'Processing...',
            data: flowData.data,
            status: flowData.status,
            timestamp: flowData.timestamp || new Date().toISOString()
          });
          return;
        }

        // Check if this is a lineage tree update
        if (message.startsWith('LINEAGE_TREE:')) {
          const lineageData = JSON.parse(message.replace('LINEAGE_TREE:', ''));
          socket.emit('lineage_tree', {
            nodes: lineageData.nodes,
            edges: lineageData.edges,
            timestamp: new Date().toISOString()
          });
          return;
        }

        // Check if this is a node status update (legacy format)
        if (message.startsWith('NODE_STATUS:')) {
          const statusData = JSON.parse(message.replace('NODE_STATUS:', ''));
          socket.emit('node_status_update', {
            nodeId: statusData.nodeId,
            status: statusData.status,
            message: statusData.message,
            timestamp: new Date().toISOString()
          });
          return;
        }

        // Determine if this is a final response
        const isFinalResponse = parsedMessage.title === 'Final Summary' || 
                               parsedMessage.type === 'final_summary' ||
                               message.includes('Root Cause Analysis Complete');

        if (isFinalResponse) {
          socket.emit('final_response', {
            ...parsedMessage,
            timestamp: new Date().toISOString()
          });
        } else {
          socket.emit('process_step', {
            ...parsedMessage,
            timestamp: new Date().toISOString()
          });
        }

      } catch (error) {
        console.error('Error processing Python output:', error);
        socket.emit('process_step', {
          title: 'Processing Error',
          type: 'error',
          content: `Error processing output: ${message}`,
          timestamp: new Date().toISOString()
        });
      }
    }

    // Handle Python script stderr output
    pythonProcess.stderr.on('data', (data) => {
      const errorMessage = data.toString();
      console.error('Python stderr:', errorMessage);
      socket.emit('process_step', {
        title: 'Python Error',
        type: 'error',
        content: errorMessage,
        timestamp: new Date().toISOString()
      });
    });

    // Handle Python script completion
    pythonProcess.on('close', (code, signal) => {
      // Process may return undefined for code and signal in some cases
      const exitCode = code !== undefined ? code : 'unknown';
      const exitSignal = signal !== undefined ? signal : 'none';
      
      console.log(`Python script finished with code: ${exitCode}, signal: ${exitSignal}`);
      if (code === undefined && signal === undefined) {
        console.log('Note: undefined exit code/signal can occur when process completes successfully');
      }
      
      // Clean up the process reference
      processManager.killProcess(socket.id);
      
      // Determine if the process completed successfully
      const isSuccess = (code === 0 || code === undefined) && !signal;
      
      if (isSuccess) {
        socket.emit('process_step', {
          title: 'Analysis Complete',
          type: 'status',
          content: 'Root Cause Analysis completed successfully.',
          timestamp: new Date().toISOString()
        });
        
        // Emit final response to stop processing state
        socket.emit('final_response', {
          content: 'RCA process completed. Review the analysis results above.',
          timestamp: new Date().toISOString()
        });
      } else if (signal) {
        socket.emit('process_step', {
          title: 'Analysis Terminated',
          type: 'warning',
          content: `Process was terminated with signal: ${signal}`,
          timestamp: new Date().toISOString()
        });
      } else {
        const errorCode = code !== undefined ? code : 'unknown';
        socket.emit('process_step', {
          title: 'Analysis Error',
          type: 'error',
          content: `Process completed with error code: ${errorCode}`,
          timestamp: new Date().toISOString()
        });
      }
    });

    // Handle Python script errors
    pythonProcess.on('error', (error) => {
      console.error('Python script error:', error);
      processManager.killProcess(socket.id);
      
      socket.emit('process_step', {
        title: 'Script Error',
        type: 'error',
        content: `Python script error: ${error.message}`,
        timestamp: new Date().toISOString()
      });
    });

  } catch (error) {
    console.error('Error starting Python process:', error);
    socket.emit('process_step', {
      title: 'Startup Error',
      type: 'error',
      content: `Failed to start RCA process: ${error.message}`,
      timestamp: new Date().toISOString()
    });
  }
}

// Health check endpoint with enhanced information
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    activeProcesses: processManager.processes.size,
    memoryUsage: process.memoryUsage(),
    uptime: process.uptime(),
    version: process.version
  });
});

// Process management endpoints for debugging (development only)
if (process.env.NODE_ENV === 'development') {
  app.get('/api/debug/processes', (req, res) => {
    res.json({
      activeProcesses: Array.from(processManager.processes.keys()),
      processCount: processManager.processes.size,
      bufferStates: Array.from(connectionBuffers.entries()).map(([id, buffer]) => ({
        socketId: id,
        position: buffer.position,
        overflow: buffer.overflow
      }))
    });
  });
}

// Process timeout handler
processManager.on('processTimeout', (socketId) => {
  console.log(`Process timeout for socket ${socketId}`);
  // Could emit a timeout message to the client here if needed
});

// Start the server
const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`🚀 Agentic DQ Backend Server running on port ${PORT}`);
  console.log(`📡 WebSocket server ready for connections`);
  console.log(`🔗 Frontend can connect to: http://localhost:${PORT}`);
  console.log(`🛡️  Security features: Rate limiting, Input validation, Process management`);
});

// Enhanced graceful shutdown
const gracefulShutdown = (signal) => {
  console.log(`Received ${signal}. Shutting down server gracefully...`);
  
  // Stop accepting new connections
  server.close(() => {
    console.log('HTTP server closed.');
    
    // Clean up all resources
    processManager.cleanup();
    connectionBuffers.clear();
    rateLimitMap.clear();
    
    console.log('All resources cleaned up. Server shutdown complete.');
    process.exit(0);
  });
  
  // Force exit after 30 seconds
  setTimeout(() => {
    console.error('Forced server shutdown after 30 seconds');
    process.exit(1);
  }, 30000);
};

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
  gracefulShutdown('UNCAUGHT_EXCEPTION');
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled rejection at:', promise, 'reason:', reason);
  gracefulShutdown('UNHANDLED_REJECTION');
});

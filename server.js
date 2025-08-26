import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { PythonShell } from 'python-shell';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Initialize Express app
const app = express();
const server = createServer(app);

// Setup Socket.IO with CORS
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(cors({ origin: "*" }));
app.use(express.json());

// Store active Python processes
const activePythonProcesses = new Map();

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log(`Client connected: ${socket.id}`);

  socket.on('disconnect', (reason) => {
    console.log(`Client disconnected: ${socket.id} | Reason: ${reason}`);
    
    // Clean up any active Python processes for this socket
    if (activePythonProcesses.has(socket.id)) {
      activePythonProcesses.get(socket.id).kill();
      activePythonProcesses.delete(socket.id);
    }
  });

  // Handle start process request from frontend
  socket.on('start_process', (userData) => {
    console.log(`Starting RCA process:`, userData);
    
    // Simple duplicate check
    if (activePythonProcesses.has(socket.id)) {
      socket.emit('error', {
        type: 'error',
        message: 'A process is already running for this connection.',
        timestamp: new Date().toISOString()
      });
      return;
    }
    
    // Extract the actual message data if it's wrapped
    const actualData = userData?.message || userData;
    startRCAProcess(socket, actualData);
  });
});

// HTTP POST endpoint for starting RCA
app.post('/api/start-rca', (req, res) => {
  console.log('RCA process started via HTTP:', req.body);
  io.emit('process_started', req.body);
  res.json({ status: 'RCA process started', data: req.body });
});

// Function to start the Python RCA process
function startRCAProcess(socket, userData) {
  try {
    console.log(`Starting Python RCA process...`);

    // Python script path
    const pythonScriptPath = path.join(__dirname, '..', 'adq-python-script', 'scripts', 'adq_agents.py');
    const venvPythonPath = path.join(__dirname, '..', 'adq-python-script', '.venv', 'Scripts', 'python.exe');
    
    // Convert userData to base64
    const userDataJson = JSON.stringify(userData);
    const userDataBase64 = Buffer.from(userDataJson, 'utf8').toString('base64');
    
    // Python options
    const options = {
      mode: 'text',
      pythonPath: venvPythonPath,
      pythonOptions: ['-u'],
      scriptPath: path.dirname(pythonScriptPath),
      args: ['--base64', userDataBase64]
    };

    // Start the Python process
    const pythonProcess = new PythonShell(path.basename(pythonScriptPath), options);
    activePythonProcesses.set(socket.id, pythonProcess);
    
    console.log(`Python process started. Active processes: ${activePythonProcesses.size}`);

    // Process timeout (5 minutes)
    const processTimeout = setTimeout(() => {
      console.log(`Killing Python process due to timeout`);
      if (activePythonProcesses.has(socket.id)) {
        pythonProcess.kill();
        activePythonProcesses.delete(socket.id);
        socket.emit('error', {
          type: 'error',
          message: 'Process timed out. Please try again.',
          timestamp: new Date().toISOString()
        });
      }
    }, 5 * 60 * 1000);

    // Send acknowledgment
    socket.emit('acknowledgment', {
      type: 'acknowledgment',
      content: 'Starting Agentic Data Quality analysis...',
      timestamp: new Date().toISOString()
    });

    // Handle Python script output
    pythonProcess.on('message', (message) => {
      try {
        console.log('Python output:', message);
        processMessage(message, socket);
      } catch (error) {
        console.error('Error processing Python output:', error);
        socket.emit('error', {
          type: 'error',
          message: `Error processing output: ${error.message}`,
          timestamp: new Date().toISOString()
        });
      }
    });

    // Handle stderr for Python errors
    pythonProcess.stderr?.on('data', (data) => {
      console.error(`Python stderr:`, data.toString());
      socket.emit('error', {
        type: 'error',
        message: `Python error: ${data.toString()}`,
        timestamp: new Date().toISOString()
      });
    });

    // Function to process messages
    function processMessage(messageStr, socket) {
      try {
        const parsedMessage = JSON.parse(messageStr);
        
        // Handle different message types
        if (parsedMessage.type === 'LINEAGE_TREE') {
          socket.emit('lineage_tree', parsedMessage);
        } else if (parsedMessage.type === 'NODE_STATUS') {
          socket.emit('node_status', parsedMessage);
        } else if (parsedMessage.type === 'FLOW_UPDATE') {
          socket.emit('process_step', {
            title: parsedMessage.data?.stepId || 'Process Update',
            type: 'flow_update',
            content: parsedMessage.data?.message || 'Processing...',
            data: parsedMessage.data,
            status: parsedMessage.data?.status,
            timestamp: parsedMessage.data?.timestamp || new Date().toISOString()
          });
        } else if (parsedMessage.title && parsedMessage.content) {
          // Handle standard message format
          const isFinalResponse = parsedMessage.title === 'Final Summary' || 
                                 parsedMessage.type === 'final_summary' ||
                                 parsedMessage.type === 'completion';

          if (isFinalResponse) {
            socket.emit('final_response', {
              type: 'final_response',
              content: parsedMessage.content,
              timestamp: parsedMessage.timestamp || new Date().toISOString()
            });
          } else if (parsedMessage.type === 'error') {
            socket.emit('error', {
              type: 'error',
              message: parsedMessage.content,
              timestamp: parsedMessage.timestamp || new Date().toISOString()
            });
          } else {
            socket.emit('process_step', {
              type: 'process_step',
              content: parsedMessage.content,
              title: parsedMessage.title,
              timestamp: parsedMessage.timestamp || new Date().toISOString()
            });
          }
        } else {
          // Fallback for unstructured messages
          socket.emit('process_step', {
            type: 'process_step',
            content: messageStr,
            timestamp: new Date().toISOString()
          });
        }
      } catch (error) {
        console.error('Error parsing message:', error);
        socket.emit('process_step', {
          type: 'process_step',
          content: messageStr,
          timestamp: new Date().toISOString()
        });
      }
    }

    // Handle Python script completion
    pythonProcess.on('close', (code) => {
      console.log(`Python script finished with code: ${code}`);
      
      clearTimeout(processTimeout);
      activePythonProcesses.delete(socket.id);
      console.log(`Process cleaned up. Active processes: ${activePythonProcesses.size}`);
      
      if (code === 0 || code === null) {
        socket.emit('final_response', {
          type: 'final_response',
          content: 'Root Cause Analysis completed successfully.',
          timestamp: new Date().toISOString()
        });
      } else {
        socket.emit('error', {
          type: 'error',
          message: `Process completed with error code: ${code}`,
          timestamp: new Date().toISOString()
        });
      }
    });

    // Handle Python script errors
    pythonProcess.on('error', (error) => {
      console.error(`Python script error:`, error);
      clearTimeout(processTimeout);
      
      socket.emit('error', {
        type: 'error',
        message: `Python script error: ${error.message}`,
        timestamp: new Date().toISOString()
      });
      
      activePythonProcesses.delete(socket.id);
    });

  } catch (error) {
    console.error(`Error starting Python process:`, error);
    
    socket.emit('error', {
      type: 'error',
      message: `Failed to start RCA process: ${error.message}`,
      timestamp: new Date().toISOString()
    });
  }
}

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    activeProcesses: activePythonProcesses.size,
    connections: io.engine.clientsCount
  });
});

// Start the server
const PORT = process.env.PORT || 3001;
server.listen(PORT, '127.0.0.1', () => {
  console.log(`🚀 Agentic DQ Backend Server running on http://127.0.0.1:${PORT}`);
  console.log(`📡 WebSocket server ready for connections`);
  console.log(`🔗 Frontend should connect to: http://127.0.0.1:${PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Shutting down server...');
  
  // Kill all active Python processes
  activePythonProcesses.forEach((process) => {
    process.kill();
  });
  
  server.close(() => {
    console.log('Server closed.');
    process.exit(0);
  });
});

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
    console.log(`[${socket.id}] Starting RCA process for:`, userData);
    
    // Check if process is already running for this socket
    if (activePythonProcesses.has(socket.id)) {
      console.log(`[${socket.id}] Process already running for this socket, ignoring duplicate request`);
      return;
    }

    // Extract and parse the actual data from the wrapper
    let actualData = userData?.message || userData;
    
    // If actualData is a string, parse it as JSON
    if (typeof actualData === 'string') {
      try {
        actualData = JSON.parse(actualData);
        console.log(`[${socket.id}] Parsed JSON string to object:`, actualData);
      } catch (e) {
        console.error(`[${socket.id}] Failed to parse JSON string:`, e);
        socket.emit('error', {
          type: 'error',
          message: 'Invalid JSON data received',
          timestamp: new Date().toISOString()
        });
        return;
      }
    }

    console.log(`[${socket.id}] Extracted message data:`, actualData);

    // Convert to the exact format expected by Python LangGraph workflow
    const pythonInputData = {
      failed_table: actualData.failed_table || "",
      failed_column: actualData.failed_column || "",
      validation_query: actualData.validation_query || "",
      execution_date: actualData.execution_date || "",
      db_type: actualData.db_type || "GCP",
      sd_threshold: actualData.sd_threshold || 3,
      expected_std_dev: actualData.expected_std_dev || 0,
      expected_value: actualData.expected_value || 0,
      actual_value: actualData.actual_value || 0,
      // Add any other fields the Python workflow expects
      issue_summary_result: null,
      data_validation_message: null,
      lineage_tree: null,
      paths_to_process: null,
      analysis_results: null,
      agent_input: actualData, // Keep original data as agent_input (now properly parsed)
      anamoly_node_response: ""
    };

    console.log(`[${socket.id}] Converted data for Python:`, pythonInputData);

    startRCAProcess(socket, pythonInputData);
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
    console.log(`[${socket.id}] Starting Python RCA process...`);

    // Python script path
    const pythonScriptPath = path.join(__dirname, '..', 'adq-python-script', 'scripts', 'adq_agents.py');
    const venvPythonPath = path.join(__dirname, '..', 'adq-python-script', '.venv', 'Scripts', 'python.exe');
    
    // Send JSON directly - no base64 conversion needed
    const userDataJson = JSON.stringify(userData);
    
    // Python options - optimized for performance and reliability
    const options = {
      mode: 'text',
      pythonPath: venvPythonPath,
      pythonOptions: ['-u', '-W', 'ignore'],
      scriptPath: path.dirname(pythonScriptPath),
      args: ['--json', userDataJson], // Changed from --base64 to --json
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
        PYTHONLEGACYWINDOWSSTDIO: '1',
        PYTHONUNBUFFERED: '1',
        PYTHONUTF8: '1',
        PYTHONDONTWRITEBYTECODE: '1' // Prevent .pyc file generation
      }
    };

    // Start the Python process
    const pythonProcess = new PythonShell(path.basename(pythonScriptPath), options);
    activePythonProcesses.set(socket.id, pythonProcess);
    
    console.log(`[${socket.id}] Python process started. Active processes: ${activePythonProcesses.size}`);

    // Process timeout (5 minutes)
    const processTimeout = setTimeout(() => {
      console.log(`[${socket.id}] Killing Python process due to timeout`);
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

    // Handle Python script output with simple line-by-line JSON parsing
    let lineBuffer = '';

    pythonProcess.on('message', (message) => {
      try {
        console.log(`[${socket.id}] Python output:`, message);
        
        // Accumulate data in line buffer
        lineBuffer += message;
        
        // Process complete lines (JSON messages)
        let newlineIndex;
        while ((newlineIndex = lineBuffer.indexOf('\n')) !== -1) {
          const completeLine = lineBuffer.substring(0, newlineIndex).trim();
          lineBuffer = lineBuffer.substring(newlineIndex + 1);
          
          // Skip empty lines
          if (!completeLine) {
            continue;
          }
          
          console.log(`[${socket.id}] Processing complete line: ${completeLine}`);
          
          // Process the complete JSON message
          processMessage(completeLine, socket);
        }
        
      } catch (error) {
        console.error(`[${socket.id}] Error processing Python output:`, error);
        socket.emit('error', {
          type: 'error',
          message: `Error processing output: ${error.message}`,
          timestamp: new Date().toISOString()
        });
      }
    });

    // Handle stderr for Python errors
    pythonProcess.stderr?.on('data', (data) => {
      console.error(`[${socket.id}] Python stderr:`, data.toString());
      socket.emit('error', {
        type: 'error',
        message: `Python error: ${data.toString()}`,
        timestamp: new Date().toISOString()
      });
    });

    // Function to process JSON messages (one per line)
    function processMessage(messageStr, socket) {
      try {
        console.log(`[${socket.id}] Processing JSON message:`, messageStr);
        
        // Skip empty messages
        if (!messageStr.trim()) {
          console.log(`[${socket.id}] Skipping empty message`);
          return;
        }
        
        const parsedMessage = JSON.parse(messageStr);
        
        // Handle different message types
        if (parsedMessage.type === 'LINEAGE_TREE') {
          socket.emit('lineage_tree', parsedMessage);
        } else if (parsedMessage.type === 'NODE_STATUS') {
          socket.emit('node_status', parsedMessage);
        } else if (parsedMessage.type === 'step_result') {
          // Handle 3-step workflow results
          socket.emit('process_step', {
            type: 'step_result',
            step_number: parsedMessage.step_number,
            title: parsedMessage.title,
            content: parsedMessage.content,
            data: parsedMessage.data,
            timestamp: parsedMessage.timestamp
          });
        } else if (parsedMessage.type === 'node_response') {
          // Handle real-time node responses
          socket.emit('process_step', {
            type: 'node_response',
            title: parsedMessage.title,
            content: parsedMessage.content,
            node_name: parsedMessage.node_name,
            data: parsedMessage.data,
            timestamp: parsedMessage.timestamp
          });
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
        console.error(`[${socket.id}] Error parsing JSON message:`, error);
        console.error(`[${socket.id}] Problematic message:`, messageStr);
        
        // Don't send malformed messages to the frontend
        console.log(`[${socket.id}] Skipping malformed JSON message due to parse error`);
      }
    }

    // Handle Python script completion
    pythonProcess.on('close', (code, signal) => {
      console.log(`[${socket.id}] Python script finished with code: ${code}, signal: ${signal}`);
      
      clearTimeout(processTimeout);
      activePythonProcesses.delete(socket.id);
      console.log(`[${socket.id}] Process cleaned up. Active processes: ${activePythonProcesses.size}`);
      
      if (code === 0 || (code === null && signal === null)) {
        // Success case - either explicit 0 or normal termination
        socket.emit('final_response', {
          type: 'final_response',
          content: 'Root Cause Analysis completed successfully.',
          timestamp: new Date().toISOString()
        });
      } else if (signal) {
        // Process was killed by signal
        socket.emit('error', {
          type: 'error',
          message: `Process was terminated (signal: ${signal}). Please try again.`,
          timestamp: new Date().toISOString()
        });
      } else if (code === null || code === undefined) {
        // Handle undefined exit code - likely successful completion on Windows
        socket.emit('final_response', {
          type: 'final_response',
          content: 'Root Cause Analysis completed successfully.',
          timestamp: new Date().toISOString()
        });
      } else {
        // Non-zero exit code
        socket.emit('error', {
          type: 'error',
          message: `Process completed with error code: ${code}`,
          timestamp: new Date().toISOString()
        });
      }
    });

    // Handle Python script errors
    pythonProcess.on('error', (error) => {
      console.error(`[${socket.id}] Python script error:`, error);
      clearTimeout(processTimeout);
      
      socket.emit('error', {
        type: 'error',
        message: `Python script error: ${error.message}`,
        timestamp: new Date().toISOString()
      });
      
      activePythonProcesses.delete(socket.id);
    });

  } catch (error) {
    console.error(`[${socket.id}] Error starting Python process:`, error);
    
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

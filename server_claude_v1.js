import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { spawn } from 'child_process';
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
      const process = activePythonProcesses.get(socket.id);
      if (process && !process.killed) {
        process.kill('SIGTERM');
      }
      activePythonProcesses.delete(socket.id);
    }
  });

  // Handle acknowledgment message from frontend
  socket.on('acknowledgment', (userData) => {
    console.log(`[${socket.id}] Received acknowledgment message:`, userData);
    socket.emit('acknowledgment', {
      type: 'acknowledgment',
      content: `${userData?.content} received! Ready to start RCA process.`,
      timestamp: new Date().toISOString()
    });
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
    let actualData = userData?.userMessage?.content || userData;

    startRCAProcess(socket, actualData);
  });

  // Handle process termination request
  socket.on('terminate_process', () => {
    console.log(`[${socket.id}] Termination requested`);
    if (activePythonProcesses.has(socket.id)) {
      const process = activePythonProcesses.get(socket.id);
      if (process && !process.killed) {
        process.kill('SIGTERM');
        activePythonProcesses.delete(socket.id);
        socket.emit('process_terminated', {
          type: 'info',
          content: 'RCA process terminated successfully.',
          timestamp: new Date().toISOString()
        });
      }
    }
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

    // Python script path - using the streaming version
    const pythonScriptPath = path.join(__dirname, '..', 'adq-python-script', 'scripts', 'streaming_agentic_dq.py');
    const venvPythonPath = path.join(__dirname, '..', 'adq-python-script', '.venv', 'Scripts', 'python.exe');
    
    // Spawn Python process
    const pythonProcess = spawn(venvPythonPath, [pythonScriptPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: path.dirname(pythonScriptPath)
    });

    // Store the process for cleanup
    activePythonProcesses.set(socket.id, pythonProcess);
    
    console.log(`[${socket.id}] Python process started. Active processes: ${activePythonProcesses.size}`);

    // Process timeout (10 minutes for complex RCA)
    const processTimeout = setTimeout(() => {
      console.log(`[${socket.id}] Killing Python process due to timeout`);
      if (activePythonProcesses.has(socket.id)) {
        pythonProcess.kill('SIGTERM');
        activePythonProcesses.delete(socket.id);
        socket.emit('error', {
          type: 'error',
          message: 'Process timed out after 10 minutes. Please try again.',
          timestamp: new Date().toISOString()
        });
      }
    }, 10 * 60 * 1000);

    // Send initial acknowledgment
    socket.emit('acknowledgment', {
      type: 'acknowledgment',
      content: 'Starting Agentic Data Quality analysis...',
      timestamp: new Date().toISOString()
    });

    // Send input data to Python process
    pythonProcess.stdin.write(JSON.stringify(userData));
    pythonProcess.stdin.end();

    // Handle Python script output with line-by-line JSON parsing
    let lineBuffer = '';

    pythonProcess.stdout?.on('data', (data) => {
      lineBuffer += data.toString();
      
      // Process complete lines
      const lines = lineBuffer.split('\n');
      lineBuffer = lines.pop() || ''; // Keep incomplete line in buffer
      
      for (const line of lines) {
        const trimmedLine = line.trim();
        if (!trimmedLine) continue;
        
        try {
          const jsonMessage = JSON.parse(trimmedLine);
          console.log(`[${socket.id}] Received JSON:`, jsonMessage);
          
          // Route different message types appropriately
          switch (jsonMessage.type) {
            case 'progress':
              socket.emit('progress', jsonMessage);
              break;
            case 'analysis_result':
              socket.emit('analysis_result', jsonMessage);
              break;
            case 'table_data':
              socket.emit('table_data', jsonMessage);
              break;
            case 'lineage_graph_structure':
              socket.emit('lineage_graph', jsonMessage);
              break;
            case 'node_status_update':
              socket.emit('node_status_update', jsonMessage);
              break;
            case 'final_report':
              socket.emit('final_report', jsonMessage);
              break;
            case 'bot':
            case 'user':
              socket.emit('message', jsonMessage);
              break;
            case 'error':
              socket.emit('error', jsonMessage);
              break;
            default:
              // Generic message handling for any other types
              socket.emit('partialResult', jsonMessage);
          }
        } catch (parseError) {
          console.error(`[${socket.id}] JSON parse error for line: "${trimmedLine}"`, parseError);
          // Don't emit parsing errors to frontend to avoid spam
        }
      }
    });

    // Handle stderr for Python errors
    pythonProcess.stderr?.on('data', (data) => {
      const errorOutput = data.toString();
      console.error(`[${socket.id}] Python stderr:`, errorOutput);
      
      // Only emit significant errors, filter out warnings
      if (errorOutput.includes('ERROR') || errorOutput.includes('CRITICAL')) {
        socket.emit('error', {
          type: 'error',
          message: `Python error: ${errorOutput}`,
          timestamp: new Date().toISOString()
        });
      }
    });

    // Handle Python script completion
    pythonProcess.on('close', (code, signal) => {
      console.log(`[${socket.id}] Python script finished with code: ${code}, signal: ${signal}`);
      
      clearTimeout(processTimeout);
      activePythonProcesses.delete(socket.id);
      console.log(`[${socket.id}] Process cleaned up. Active processes: ${activePythonProcesses.size}`);
      
      if (code === 0) {
        socket.emit('process_completed', {
          type: 'completion',
          content: 'RCA analysis completed successfully.',
          timestamp: new Date().toISOString()
        });
      } else if (code !== null && code !== 0) {
        socket.emit('error', {
          type: 'error',
          message: `Python script exited with error code: ${code}`,
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

    // Handle process exit
    pythonProcess.on('exit', (code, signal) => {
      console.log(`[${socket.id}] Python process exited with code: ${code}, signal: ${signal}`);
      clearTimeout(processTimeout);
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
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    activeProcesses: activePythonProcesses.size
  });
});

// Get active processes count
app.get('/api/status', (req, res) => {
  res.json({ 
    activeProcesses: activePythonProcesses.size,
    timestamp: new Date().toISOString()
  });
});

// Start the server
const PORT = process.env.PORT || 3001;
const HOST = process.env.HOST || '127.0.0.1';

server.listen(PORT, HOST, () => {
  console.log(`🚀 Agentic DQ Backend Server running on http://${HOST}:${PORT}`);
  console.log(`📡 WebSocket server ready for connections`);
  console.log(`🔗 Frontend should connect to: http://${HOST}:${PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', handleShutdown);
process.on('SIGINT', handleShutdown);

function handleShutdown() {
  console.log('Shutting down server...');
  
  // Kill all active Python processes
  activePythonProcesses.forEach((process, socketId) => {
    console.log(`Terminating Python process for socket: ${socketId}`);
    if (process && !process.killed) {
      process.kill('SIGTERM');
    }
  });
  
  activePythonProcesses.clear();
  
  server.close(() => {
    console.log('Server closed.');
    process.exit(0);
  });
}
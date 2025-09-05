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

// Store active sessions and conversations
const activeProcesses = new Map();
const conversations = new Map();
const userSessions = new Map();

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log(`Client connected: ${socket.id}`);

  // Initialize user session
  userSessions.set(socket.id, {
    conversations: [],
    currentConversation: null
  });

  socket.on('disconnect', (reason) => {
    console.log(`Client disconnected: ${socket.id} | Reason: ${reason}`);
    
    // Clean up active processes
    if (activeProcesses.has(socket.id)) {
      const process = activeProcesses.get(socket.id);
      if (process && !process.killed) {
        process.kill('SIGTERM');
      }
      activeProcesses.delete(socket.id);
    }
    
    // Clean up session
    userSessions.delete(socket.id);
  });

  // Handle new conversation creation
  socket.on('create_new_conversation', (data) => {
    const conversationId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const conversation = {
      id: conversationId,
      title: data.title || 'New Conversation',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      messages: [],
      type: data.type || 'chat' // 'chat' or 'rca'
    };

    conversations.set(conversationId, conversation);
    
    const userSession = userSessions.get(socket.id);
    userSession.conversations.unshift(conversation);
    userSession.currentConversation = conversationId;

    socket.emit('conversation_created', {
      conversation: conversation,
      conversations: userSession.conversations
    });
  });

  // Handle conversation selection
  socket.on('select_conversation', (data) => {
    const { conversationId } = data;
    const conversation = conversations.get(conversationId);
    
    if (conversation) {
      const userSession = userSessions.get(socket.id);
      userSession.currentConversation = conversationId;
      
      socket.emit('conversation_selected', {
        conversation: conversation,
        messages: conversation.messages
      });
    }
  });

  // Handle regular chat messages
  socket.on('send_chat_message', async (data) => {
    const { message, conversationId } = data;
    const conversation = conversations.get(conversationId);
    
    if (!conversation) {
      socket.emit('error', { message: 'Conversation not found' });
      return;
    }

    // Add user message to conversation
    const userMessage = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: message,
      type: 'normal',
      status: 'static',
      timestamp: new Date().toISOString(),
    };
    
    conversation.messages.push(userMessage);
    conversation.updated_at = new Date().toISOString();
    // Emit message immediately
    socket.emit('message_received', {
      conversationId,
      message: userMessage
    });

    // Check if this is an RCA request
    if (isRCARequest(message)) {
      // Update conversation type and title
      conversation.type = 'rca';
      conversation.title = extractRCATitle(message);
      
      // Start RCA process
      startRCAProcess(socket, conversationId, message);
    } else {
      // Handle regular chat with simulated AI response
      setTimeout(() => {
        const aiResponse = {
          id: `msg_${Date.now()}`,
          content: generateChatResponse(message),
          role: 'bot',
          type: 'normal',
          status: 'static',
          timestamp: new Date().toISOString(),
        };
        
        conversation.messages.push(aiResponse);
        conversation.updated_at = new Date().toISOString();
        
        socket.emit('message_received', {
          conversationId,
          message: aiResponse
        });
      }, 1000);
    }
  });

  // Handle RCA process initiation
  // socket.on('start_rca_analysis', (data) => {
  //   const { conversationId, rcaData } = data;
  //   startRCAProcess(socket, conversationId, rcaData);
  // });

  // Handle process termination
  socket.on('terminate_process', (data) => {
    const { conversationId } = data;
    const processKey = `${socket.id}_${conversationId}`;
    
    if (activeProcesses.has(processKey)) {
      const process = activeProcesses.get(processKey);
      if (process && !process.killed) {
        process.kill('SIGTERM');
        activeProcesses.delete(processKey);
        
        socket.emit('process_terminated', {
          conversationId,
          message: 'RCA process terminated successfully.'
        });
      }
    }
  });

  // Get conversation history
  socket.on('get_conversations', () => {
    const userSession = userSessions.get(socket.id);
    socket.emit('conversations_list', {
      conversations: userSession.conversations
    });
  });
});

// Function to detect RCA requests
function isRCARequest(message) {
  const rcaKeywords = [
    'investigate', 'deviation', 'anomaly', 'root cause', 'analyze', 'rca',
    'data quality', 'lineage', 'trace', 'failed_rule', 'validation'
  ];
  
  const lowerMessage = message.toLowerCase();
  return rcaKeywords.some(keyword => lowerMessage.includes(keyword)) ||
         message.includes('{') && message.includes('}'); // JSON-like structure
}

// Function to extract RCA title from message
function extractRCATitle(message) {
  if (message.length > 50) {
    return message.substring(0, 47) + '...';
  }
  return message;
}

// Function to generate chat responses
function generateChatResponse(message) {
  const responses = [
    "I understand your question. How can I help you further?",
    "That's an interesting point. Could you provide more details?",
    "I'm here to assist you. What specific information are you looking for?",
    "Let me help you with that. Can you clarify what you need?",
    "I see what you're asking. Here's what I can tell you...",
  ];
  
  return responses[Math.floor(Math.random() * responses.length)];
}

// Function to start RCA process
function startRCAProcess(socket, conversationId, inputData) {
  const processKey = `${socket.id}_${conversationId}`;
  
  // Check if process is already running
  if (activeProcesses.has(processKey)) {
    console.log(`RCA process already running for ${processKey}`);
    return;
  }

  try {
    console.log(`Starting RCA process for conversation: ${conversationId}`);
    
    // Python script path - using the updated version
    const pythonScriptPath = path.join(__dirname, '..', 'adq-python-script', 'scripts', 'agentic_dq_claude_v2.py');
    const venvPythonPath = path.join(__dirname, '..', 'adq-python-script', '.venv', 'Scripts', 'python.exe');

    // Parse input data
    let rcaInput;
    try {
      // Try to parse as JSON first
      rcaInput = JSON.parse(inputData);
      console.log("Parsed RCA input:", rcaInput);
    } catch (e) {
      // If not JSON, create structured input from text
      rcaInput = {
        failed_table: "target_table",
        failed_column: "data_quality_metric",
        db_type: "GCP",
        validation_query: inputData,
        agent_input: inputData
      };
    }

    // Spawn Python process
    const pythonProcess = spawn(venvPythonPath, [pythonScriptPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: path.dirname(pythonScriptPath)
    });

    activeProcesses.set(processKey, pythonProcess);
    console.log(`Python process started for ${processKey}`);

    // Process timeout (10 minutes)
    const processTimeout = setTimeout(() => {
      console.log(`Killing Python process due to timeout: ${processKey}`);
      if (activeProcesses.has(processKey)) {
        pythonProcess.kill('SIGTERM');
        activeProcesses.delete(processKey);
        
        socket.emit('rca_error', {
          conversationId,
          message: 'RCA process timed out after 10 minutes. Please try again.',
          timestamp: new Date().toISOString()
        });
      }
    }, 10 * 60 * 1000);

    // Send input data to Python process
    pythonProcess.stdin.write(JSON.stringify(rcaInput));
    pythonProcess.stdin.end();

    // Handle Python output with line-by-line processing
    let lineBuffer = '';

    pythonProcess.stdout?.on('data', (data) => {
      lineBuffer += data.toString();
      
      const lines = lineBuffer.split('\n');
      lineBuffer = lines.pop() || '';
      
      for (const line of lines) {
        const trimmedLine = line.trim();
        if (!trimmedLine) continue;
        
        try {
          const jsonMessage = JSON.parse(trimmedLine);
          console.log(`[${processKey}] Received JSON:`, jsonMessage);
          
          // Route different message types appropriately
          switch (jsonMessage.role) {
            case 'bot':
            case 'user':
              socket.emit('message', jsonMessage);
              break;
            case 'error':
              // socket.emit('error', jsonMessage);
              console.error('Python error (suppressed from frontend):', jsonMessage);
              break;
            default:
              // Store in conversation and emit generic update
              const conversation = conversations.get(conversationId);
              if (conversation) {
                const rcaMessage = {
                  id: `msg_${Date.now()}_${Math.random()}`,
                  sender: 'ai',
                  timestamp: new Date().toISOString(),
                  type: jsonMessage.type || 'rca_update',
                  content: jsonMessage.content || jsonMessage.message || '',
                  metadata: jsonMessage.metadata || jsonMessage.data || {}
                };

                conversation.messages.push(rcaMessage);
                conversation.updated_at = new Date().toISOString();

                socket.emit('rca_update', {
                  conversationId,
                  message: rcaMessage,
                  rawData: jsonMessage
                });
              }
          }
          
        } catch (parseError) {
          // Improved error handling with more context
          console.error(`[${processKey}] JSON parse error:`, parseError.message);
          console.error(`[${processKey}] Problematic line (first 200 chars):`, trimmedLine.substring(0, 200));
          
          // Check if this looks like a log line that should be ignored
          const isLogLine = trimmedLine.match(/^\d{4}-\d{2}-\d{2}|ERROR|INFO|DEBUG|WARNING|CRITICAL/);
          if (isLogLine) {
            console.log(`[${processKey}] Skipping log line:`, trimmedLine.substring(0, 100));
          } else {
            // This might be a real issue - log more details
            console.error(`[${processKey}] Unexpected non-JSON output:`, trimmedLine);
          }
          
          // Don't spam frontend with parsing errors - these are now handled gracefully
        }
      }
    });

    // Handle stderr
    pythonProcess.stderr?.on('data', (data) => {
      const errorOutput = data.toString();
      console.error(`Python stderr for ${processKey}:`, errorOutput);
      
      // if (errorOutput.includes('ERROR') || errorOutput.includes('CRITICAL')) {
      //   socket.emit('rca_error', {
      //     conversationId,
      //     message: `Python error: ${errorOutput}`,
      //     timestamp: new Date().toISOString()
      //   });
      // }
    });

    // Handle process completion
    pythonProcess.on('close', (code, signal) => {
      console.log(`Python process finished for ${processKey}: code=${code}, signal=${signal}`);
      
      clearTimeout(processTimeout);
      activeProcesses.delete(processKey);
      
      if (code === 0) {
        socket.emit('rca_completed', {
          conversationId,
          message: 'Root Cause Analysis completed successfully.',
          timestamp: new Date().toISOString()
        });
      } else if (code !== null && code !== 0) {
        socket.emit('rca_error', {
          conversationId,
          message: `RCA process failed with exit code: ${code}`,
          timestamp: new Date().toISOString()
        });
        console.error(`RCA process failed with exit code: ${code}`);
      }
    });

    // Handle process errors
    pythonProcess.on('error', (error) => {
      console.error(`Python process error for ${processKey}:`, error);
      clearTimeout(processTimeout);
      
      // socket.emit('rca_error', {
      //   conversationId,
      //   message: `Failed to start RCA process: ${error.message}`,
      //   timestamp: new Date().toISOString()
      // });
      
      activeProcesses.delete(processKey);
    });

  } catch (error) {
    console.error(`Error starting RCA process for ${processKey}:`, error);
    
    socket.emit('rca_error', {
      conversationId,
      message: `Failed to initialize RCA process: ${error.message}`,
      timestamp: new Date().toISOString()
    });
  }
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    activeProcesses: activeProcesses.size,
    activeSessions: userSessions.size,
    totalConversations: conversations.size
  });
});

// Get system status
app.get('/api/status', (req, res) => {
  res.json({
    activeProcesses: activeProcesses.size,
    activeSessions: userSessions.size,
    totalConversations: conversations.size,
    timestamp: new Date().toISOString()
  });
});

// Start server
const PORT = process.env.PORT || 3001;
const HOST = process.env.HOST || '127.0.0.1';

server.listen(PORT, HOST, () => {
  console.log(`🚀 Integrated Backend running on http://${HOST}:${PORT}`);
  console.log(`📡 WebSocket server ready for connections`);
  console.log(`🔗 Frontend should connect to: http://${HOST}:${PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', handleShutdown);
process.on('SIGINT', handleShutdown);

function handleShutdown() {
  console.log('Shutting down server...');
  
  activeProcesses.forEach((process, key) => {
    console.log(`Terminating process: ${key}`);
    if (process && !process.killed) {
      process.kill('SIGTERM');
    }
  });
  
  activeProcesses.clear();
  
  server.close(() => {
    console.log('Server closed.');
    process.exit(0);
  });
}
import { useReducer, useEffect } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import { appReducer, initialState, actionTypes } from './reducers/appReducer'
import { validateAndSanitize, messageSchema, ClientRateLimit, validateMessageSize } from './utils/validation'
import ProcessFlow from './components/ProcessFlow'
import Chatbot from './components/Chatbot'
import LineageTree from './components/LineageTree'
import ConnectionStatus from './components/ConnectionStatus'
import ErrorBoundary from './components/ErrorBoundary'
import './App.css'

function App() {
  const [state, dispatch] = useReducer(appReducer, initialState)
  const { socket, connectionState, isConnected, sendMessage, reconnect } = useWebSocket('http://localhost:3001')
  
  // Initialize rate limiter
  const rateLimiter = new ClientRateLimit(30, 60000) // 30 requests per minute

  const { 
    processSteps, 
    chatHistory, 
    isProcessing, 
    theme, 
    currentView, 
    lineageData 
  } = state

  // Enhanced theme management with system preference detection
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'system'
    dispatch({ type: actionTypes.SET_THEME, payload: savedTheme })
    applyTheme(savedTheme)

    // Listen for system theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleSystemThemeChange = () => {
      if (theme === 'system') {
        applyTheme('system')
      }
    }
    
    mediaQuery.addEventListener('change', handleSystemThemeChange)
    return () => mediaQuery.removeEventListener('change', handleSystemThemeChange)
  }, [theme])

  // Update connection status in state
  useEffect(() => {
    dispatch({ type: actionTypes.SET_CONNECTION_STATUS, payload: connectionState })
  }, [connectionState])

  const applyTheme = (newTheme) => {
    const isDark = newTheme === 'dark' || 
      (newTheme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)
    
    if (isDark) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }

  const toggleTheme = () => {
    const themes = ['system', 'light', 'dark']
    const currentIndex = themes.indexOf(theme)
    const newTheme = themes[(currentIndex + 1) % themes.length]
    
    dispatch({ type: actionTypes.SET_THEME, payload: newTheme })
    localStorage.setItem('theme', newTheme)
    applyTheme(newTheme)
  }

  const getThemeIcon = () => {
    switch (theme) {
      case 'system': return '🌓'
      case 'light': return '☀️'
      case 'dark': return '🌙'
      default: return '🌓'
    }
  }

  // Socket event listeners with enhanced error handling
  useEffect(() => {
    if (!socket) return

    const handleProcessStep = (data) => {
      console.log('Process step received:', data)
      dispatch({ type: actionTypes.ADD_PROCESS_STEP, payload: data })
    }

    const handleFinalResponse = (data) => {
      console.log('Final response received:', data)
      dispatch({ 
        type: actionTypes.ADD_CHAT_MESSAGE, 
        payload: {
          role: 'bot',
          content: data.content,
          timestamp: data.timestamp
        }
      })
      dispatch({ type: actionTypes.SET_PROCESSING, payload: false })
    }

    const handleLineageTree = (data) => {
      console.log('Lineage tree received:', data)
      dispatch({ 
        type: actionTypes.SET_LINEAGE_DATA, 
        payload: {
          nodes: data.nodes || [],
          edges: data.edges || []
        }
      })
    }

    const handleNodeStatusUpdate = (data) => {
      console.log('Node status update received:', data)
      dispatch({ 
        type: actionTypes.UPDATE_NODE_STATUS, 
        payload: {
          nodeId: data.nodeId,
          status: data.status,
          message: data.message
        }
      })
    }

    const handleError = (error) => {
      console.error('Socket error received:', error)
      dispatch({ 
        type: actionTypes.ADD_CHAT_MESSAGE, 
        payload: {
          role: 'bot',
          content: `Server error: ${error.message || 'Unknown error occurred'}`,
          timestamp: new Date().toISOString()
        }
      })
      if (error.type === 'rate_limit' || error.type === 'validation') {
        dispatch({ type: actionTypes.SET_PROCESSING, payload: false })
      }
    }

    const handleWarning = (warning) => {
      console.warn('Socket warning received:', warning)
      dispatch({ 
        type: actionTypes.ADD_CHAT_MESSAGE, 
        payload: {
          role: 'bot',
          content: `Warning: ${warning.message || 'System warning'}`,
          timestamp: new Date().toISOString()
        }
      })
    }

    // Add event listeners
    socket.on('process_step', handleProcessStep)
    socket.on('final_response', handleFinalResponse)
    socket.on('lineage_tree', handleLineageTree)
    socket.on('node_status_update', handleNodeStatusUpdate)
    socket.on('error', handleError)
    socket.on('warning', handleWarning)

    // Cleanup
    return () => {
      socket.off('process_step', handleProcessStep)
      socket.off('final_response', handleFinalResponse)
      socket.off('lineage_tree', handleLineageTree)
      socket.off('node_status_update', handleNodeStatusUpdate)
      socket.off('error', handleError)
      socket.off('warning', handleWarning)
    }
  }, [socket])

  // Handle sending a message with validation and rate limiting
  const handleSendMessage = (message) => {
    if (!isConnected || isProcessing) return

    try {
      // Rate limiting check
      if (!rateLimiter.checkLimit()) {
        const timeUntilReset = Math.ceil(rateLimiter.getTimeUntilReset() / 1000);
        dispatch({ 
          type: actionTypes.ADD_CHAT_MESSAGE, 
          payload: {
            role: 'bot',
            content: `Rate limit exceeded. Please wait ${timeUntilReset} seconds before sending another message.`,
            timestamp: new Date().toISOString()
          }
        });
        return;
      }

      // Validate message size
      validateMessageSize(message, 50000);

      // Prepare and validate message data
      const messageData = {
        message: message,
        timestamp: new Date().toISOString()
      };

      // Validate the message structure
      const validatedMessage = validateAndSanitize(messageData, messageSchema);

      // Add user message to chat history
      dispatch({ 
        type: actionTypes.ADD_CHAT_MESSAGE, 
        payload: {
          role: 'user',
          content: message,
          timestamp: new Date().toISOString()
        }
      });

      // Check if this is a JSON submission (for RCA Agent)
      const isJsonSubmission = message.trim().startsWith('{') && message.trim().endsWith('}')
      
      // Special handling for RCA Agent selection - show JSON format request
      if (message.includes("RCA Agent") && !isJsonSubmission) {
        setTimeout(() => {
          const botResponse = `Please provide the failure JSON in the below format:

\`\`\`json
{
  "failed_table": "<Fully Qualified table name>",
  "failed_column": "<Column name>", 
  "db_type": "<GCP/Teradata>",
  "validation_query": "<SQL that contains the validation>",
  "sd_threshold": "",
  "expected_std_dev": "",
  "expected_value": "",
  "actual_value": "",
  "execution_date": ""
}
\`\`\``
          
          dispatch({ 
            type: actionTypes.ADD_CHAT_MESSAGE, 
            payload: {
              role: 'bot',
              content: botResponse,
              timestamp: new Date().toISOString()
            }
          });
        }, 800);
        return;
      }

      // Clear previous process steps and lineage data, start processing for all requests including JSON
      dispatch({ type: actionTypes.CLEAR_PROCESS_DATA });
      dispatch({ type: actionTypes.SET_PROCESSING, payload: true });

      // For JSON submissions, parse and validate the JSON object
      let messageToSend = validatedMessage.message;
      if (isJsonSubmission) {
        try {
          const parsedJson = JSON.parse(message);
          // Additional validation for RCA JSON data could be added here
          messageToSend = parsedJson;
        } catch (e) {
          console.error('Invalid JSON submitted:', e);
          dispatch({ 
            type: actionTypes.ADD_CHAT_MESSAGE, 
            payload: {
              role: 'bot',
              content: 'Invalid JSON format. Please check your syntax and try again.',
              timestamp: new Date().toISOString()
            }
          });
          dispatch({ type: actionTypes.SET_PROCESSING, payload: false });
          return;
        }
      }

      // Send message to backend using the new hook
      const success = sendMessage('start_process', {
        message: messageToSend,
        timestamp: validatedMessage.timestamp
      });

      if (!success) {
        dispatch({ 
          type: actionTypes.ADD_CHAT_MESSAGE, 
          payload: {
            role: 'bot',
            content: 'Failed to send message. Please check your connection.',
            timestamp: new Date().toISOString()
          }
        });
        dispatch({ type: actionTypes.SET_PROCESSING, payload: false });
      }

    } catch (error) {
      console.error('Message validation error:', error);
      dispatch({ 
        type: actionTypes.ADD_CHAT_MESSAGE, 
        payload: {
          role: 'bot',
          content: `Input validation error: ${error.message}`,
          timestamp: new Date().toISOString()
        }
      });
      dispatch({ type: actionTypes.SET_PROCESSING, payload: false });
    }
  }

  // Handle stopping the current process
  const handleStopProcess = () => {
    if (!isConnected || !isProcessing) return;

    const success = sendMessage('stop_process');
    
    if (success) {
      dispatch({ 
        type: actionTypes.ADD_CHAT_MESSAGE, 
        payload: {
          role: 'user',
          content: '🛑 Stop requested',
          timestamp: new Date().toISOString()
        }
      });
    } else {
      dispatch({ 
        type: actionTypes.ADD_CHAT_MESSAGE, 
        payload: {
          role: 'bot',
          content: 'Failed to send stop request. Please check your connection.',
          timestamp: new Date().toISOString()
        }
      });
    }
  }

  return (
    <ErrorBoundary>
      <div className="h-screen w-screen flex transition-colors duration-300 bg-slate-50 dark:bg-slate-900">
        {/* Left Panel - Process Flow & Data Lineage */}
        <div className="flex-1 border-r flex flex-col bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700">
          {/* View Toggle */}
          <div className="px-6 py-4 border-b bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700">
            <div className="flex items-center justify-between mb-2">
              <h1 className="text-xl font-bold text-slate-800 dark:text-slate-200">
                {currentView === 'analysis' ? '📊 Analysis Workflow' : '🌳 Data Lineage'}
              </h1>
              <div className="flex bg-slate-100 dark:bg-slate-700 rounded-lg p-1 border border-slate-200 dark:border-slate-600">
                <button
                  onClick={() => dispatch({ type: actionTypes.SET_VIEW, payload: 'analysis' })}
                  className={`px-3 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                    currentView === 'analysis'
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-600'
                  }`}
                >
                  📊 Workflow
                </button>
                <button
                  onClick={() => dispatch({ type: actionTypes.SET_VIEW, payload: 'lineage' })}
                  className={`px-3 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                    currentView === 'lineage'
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-600'
                  }`}
                >
                  🌳 Lineage
                </button>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <p className="text-sm text-slate-600 dark:text-slate-400">
                {currentView === 'analysis' 
                  ? 'Live AI agent execution steps and process monitoring' 
                  : 'Interactive data lineage with real-time status updates'
                }
              </p>
              <ConnectionStatus connectionState={connectionState} onReconnect={reconnect} />
            </div>
          </div>
        
        {/* Scrollable Content Area */}
        <div className="flex-1 overflow-y-auto">
          {currentView === 'analysis' ? (
            <ProcessFlow 
              processSteps={processSteps} 
              isProcessing={isProcessing}
              theme={theme}
            />
          ) : (
            <LineageTree 
              theme={theme} 
              nodes={lineageData.nodes} 
              edges={lineageData.edges}
              nodeStatusUpdates={lineageData.nodeStatusUpdates}
            />
          )}
        </div>
      </div>

      {/* Right Panel - Chatbot with Header */}
      <div className="flex-1 flex flex-col bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700">
        {/* Chat Header with Title and Theme Toggle */}
        <div className="px-6 py-4 border-b bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white text-lg font-bold">🤖</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-800 dark:text-slate-200">
                  Agentic Data Quality
                </h1>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  AI-powered data quality analysis and root cause detection
                </p>
              </div>
            </div>
            <button 
              onClick={toggleTheme}
              className="flex items-center space-x-2 px-4 py-2 border rounded-lg shadow-sm hover:shadow-md transition-all duration-200 bg-white dark:bg-slate-700 border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-600"
              title={`Theme: ${theme} (click to cycle)`}
            >
              <span className="text-lg">{getThemeIcon()}</span>
              <span className="text-sm font-medium capitalize">{theme}</span>
            </button>
          </div>
        </div>
        
        {/* Chat Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <Chatbot 
            chatHistory={chatHistory}
            onSendMessage={handleSendMessage}
            onStopProcess={handleStopProcess}
            isProcessing={isProcessing}
            theme={theme}
          />
        </div>
      </div>
    </div>
    </ErrorBoundary>
  )
}

export default App

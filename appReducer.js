const initialState = {
  processSteps: [],
  chatHistory: [],
  isProcessing: false,
  theme: 'system',
  currentView: 'analysis',
  lineageData: { nodes: [], edges: [], nodeStatusUpdates: {} },
  connectionStatus: 'disconnected',
  error: null
};

const actionTypes = {
  SET_PROCESSING: 'SET_PROCESSING',
  ADD_PROCESS_STEP: 'ADD_PROCESS_STEP',
  ADD_CHAT_MESSAGE: 'ADD_CHAT_MESSAGE',
  SET_LINEAGE_DATA: 'SET_LINEAGE_DATA',
  UPDATE_NODE_STATUS: 'UPDATE_NODE_STATUS',
  CLEAR_PROCESS_DATA: 'CLEAR_PROCESS_DATA',
  SET_THEME: 'SET_THEME',
  SET_VIEW: 'SET_VIEW',
  SET_CONNECTION_STATUS: 'SET_CONNECTION_STATUS',
  SET_ERROR: 'SET_ERROR'
};

function appReducer(state, action) {
  switch (action.type) {
    case actionTypes.SET_PROCESSING:
      return {
        ...state,
        isProcessing: action.payload
      };

    case actionTypes.ADD_PROCESS_STEP:
      return {
        ...state,
        processSteps: [...state.processSteps, {
          ...action.payload,
          id: Date.now() + Math.random(),
          timestamp: action.payload.timestamp || new Date().toISOString()
        }]
      };
      
    case actionTypes.ADD_CHAT_MESSAGE:
      return {
        ...state,
        chatHistory: [...state.chatHistory, {
          ...action.payload,
          id: Date.now() + Math.random(),
          timestamp: action.payload.timestamp || new Date().toISOString()
        }]
      };
      
    case actionTypes.SET_LINEAGE_DATA:
      return {
        ...state,
        lineageData: {
          ...state.lineageData,
          ...action.payload
        }
      };
      
    case actionTypes.UPDATE_NODE_STATUS:
      return {
        ...state,
        lineageData: {
          ...state.lineageData,
          nodeStatusUpdates: {
            ...state.lineageData.nodeStatusUpdates,
            [action.payload.nodeId]: {
              status: action.payload.status,
              message: action.payload.message
            }
          }
        }
      };
      
    case actionTypes.CLEAR_PROCESS_DATA:
      return {
        ...state,
        processSteps: [],
        lineageData: { nodes: [], edges: [], nodeStatusUpdates: {} }
      };

    case actionTypes.SET_THEME:
      return {
        ...state,
        theme: action.payload
      };

    case actionTypes.SET_VIEW:
      return {
        ...state,
        currentView: action.payload
      };

    case actionTypes.SET_CONNECTION_STATUS:
      return {
        ...state,
        connectionStatus: action.payload
      };

    case actionTypes.SET_ERROR:
      return {
        ...state,
        error: action.payload
      };
      
    default:
      return state;
  }
}

export { appReducer, initialState, actionTypes };

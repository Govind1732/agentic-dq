import { useState, useEffect, useRef, useCallback } from 'react';
import { io } from 'socket.io-client';

const CONNECTION_STATES = {
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  RECONNECTING: 'reconnecting',
  ERROR: 'error'
};

export function useWebSocket(url, options = {}) {
  const [connectionState, setConnectionState] = useState(CONNECTION_STATES.DISCONNECTED);
  const [socket, setSocket] = useState(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = options.maxReconnectAttempts || 5;
  const reconnectDelay = options.reconnectDelay || 1000;

  useEffect(() => {
    let currentSocket = null;

    const connect = () => {
      if (currentSocket?.connected) return;

      setConnectionState(CONNECTION_STATES.CONNECTING);
      
      currentSocket = io(url, {
        transports: ['websocket'],
        timeout: 20000,
        ...(options.socketOptions || {})
      });

      currentSocket.on('connect', () => {
        console.log('WebSocket connected');
        setConnectionState(CONNECTION_STATES.CONNECTED);
        reconnectAttempts.current = 0;
        setSocket(currentSocket);
      });

      currentSocket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        setConnectionState(CONNECTION_STATES.DISCONNECTED);
        
        if (reason === 'io server disconnect') {
          return;
        }
        
        // Attempt reconnection
        if (reconnectAttempts.current < maxReconnectAttempts) {
          setConnectionState(CONNECTION_STATES.RECONNECTING);
          reconnectAttempts.current++;
          
          setTimeout(() => {
            connect();
          }, reconnectDelay * Math.pow(2, reconnectAttempts.current));
        } else {
          setConnectionState(CONNECTION_STATES.ERROR);
        }
      });

      currentSocket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        
        if (reconnectAttempts.current < maxReconnectAttempts) {
          setConnectionState(CONNECTION_STATES.RECONNECTING);
          reconnectAttempts.current++;
          
          setTimeout(() => {
            connect();
          }, reconnectDelay * Math.pow(2, reconnectAttempts.current));
        } else {
          setConnectionState(CONNECTION_STATES.ERROR);
        }
      });
    };

    connect();
    
    return () => {
      if (currentSocket) {
        currentSocket.close();
      }
    };
  }, [url, maxReconnectAttempts, reconnectDelay, options.socketOptions]);

  const sendMessage = useCallback((event, data) => {
    if (socket?.connected) {
      socket.emit(event, data);
      return true;
    }
    return false;
  }, [socket]);

  const reconnect = useCallback(() => {
    if (socket) {
      socket.close();
    }
    reconnectAttempts.current = 0;
    // The useEffect will handle reconnection
  }, [socket]);

  return {
    socket,
    connectionState,
    isConnected: connectionState === CONNECTION_STATES.CONNECTED,
    sendMessage,
    reconnect
  };
}

export { CONNECTION_STATES };

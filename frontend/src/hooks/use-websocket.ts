import { useEffect, useRef, useState, useCallback } from 'react';
import { toast } from 'react-hot-toast';
import { WebSocketMessage, PriceUpdateMessage, PositionUpdateMessage, OrderUpdateMessage, AlertMessage } from '@/types';

interface UseWebSocketOptions {
  url?: string;
  protocols?: string | string[];
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: Event) => void;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

interface WebSocketState {
  socket: WebSocket | null;
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected';
  lastMessage: WebSocketMessage | null;
  error: string | null;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    url = process.env.WS_URL || 'ws://localhost:8000/ws',
    protocols,
    onMessage,
    onError,
    onOpen,
    onClose,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
  } = options;

  const [state, setState] = useState<WebSocketState>({
    socket: null,
    isConnected: false,
    connectionStatus: 'disconnected',
    lastMessage: null,
    error: null,
  });

  const reconnectCount = useRef(0);
  const reconnectTimer = useRef<NodeJS.Timeout>();
  const shouldReconnect = useRef(true);

  const connect = useCallback(() => {
    try {
      setState(prev => ({ 
        ...prev, 
        connectionStatus: 'connecting', 
        error: null 
      }));

      const socket = new WebSocket(url, protocols);

      socket.onopen = (event) => {
        console.log('WebSocket connected');
        setState(prev => ({
          ...prev,
          socket,
          isConnected: true,
          connectionStatus: 'connected',
          error: null,
        }));
        
        reconnectCount.current = 0;
        onOpen?.(event);
        
        toast.success('Connected to live data feed', {
          duration: 2000,
        });
      };

      socket.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          setState(prev => ({
            ...prev,
            lastMessage: message,
          }));

          onMessage?.(message);
          
          // Handle specific message types
          handleMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      socket.onerror = (event) => {
        console.error('WebSocket error:', event);
        setState(prev => ({
          ...prev,
          error: 'Connection error',
        }));
        
        onError?.(event);
      };

      socket.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setState(prev => ({
          ...prev,
          socket: null,
          isConnected: false,
          connectionStatus: 'disconnected',
        }));

        onClose?.(event);

        // Attempt to reconnect if not manually closed
        if (shouldReconnect.current && reconnectCount.current < reconnectAttempts) {
          reconnectCount.current++;
          console.log(`Attempting to reconnect (${reconnectCount.current}/${reconnectAttempts})...`);
          
          reconnectTimer.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
          
          toast.error(`Connection lost. Reconnecting... (${reconnectCount.current}/${reconnectAttempts})`, {
            duration: 3000,
          });
        } else if (reconnectCount.current >= reconnectAttempts) {
          toast.error('Failed to reconnect. Please refresh the page.', {
            duration: 5000,
          });
        }
      };

    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      setState(prev => ({
        ...prev,
        error: 'Failed to create connection',
        connectionStatus: 'disconnected',
      }));
    }
  }, [url, protocols, onMessage, onError, onOpen, onClose, reconnectAttempts, reconnectInterval]);

  const disconnect = useCallback(() => {
    shouldReconnect.current = false;
    
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
    }

    if (state.socket) {
      state.socket.close();
    }
  }, [state.socket]);

  const sendMessage = useCallback((message: any) => {
    if (state.socket && state.isConnected) {
      try {
        const messageString = typeof message === 'string' ? message : JSON.stringify(message);
        state.socket.send(messageString);
        return true;
      } catch (error) {
        console.error('Error sending WebSocket message:', error);
        return false;
      }
    }
    return false;
  }, [state.socket, state.isConnected]);

  // Handle different message types
  const handleMessage = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'PRICE_UPDATE':
        // Handle price updates
        break;
        
      case 'POSITION_UPDATE':
        // Handle position updates
        break;
        
      case 'ORDER_UPDATE':
        // Handle order updates
        break;
        
      case 'ALERT':
        const alertMessage = message as AlertMessage;
        const alert = alertMessage.data;
        
        // Show toast notification for alerts
        const toastOptions = {
          duration: alert.severity >= 4 ? 10000 : 5000,
        };
        
        switch (alert.type) {
          case 'CRITICAL':
            toast.error(alert.message, toastOptions);
            break;
          case 'ERROR':
            toast.error(alert.message, toastOptions);
            break;
          case 'WARNING':
            toast((t) => (
              <div className="flex items-center space-x-2">
                <span className="text-trading-pending">⚠️</span>
                <span>{alert.message}</span>
              </div>
            ), toastOptions);
            break;
          case 'INFO':
            toast.success(alert.message, toastOptions);
            break;
        }
        break;
        
      case 'SYSTEM_STATUS':
        // Handle system status updates
        break;
        
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  // Connect on mount
  useEffect(() => {
    shouldReconnect.current = true;
    connect();

    return () => {
      shouldReconnect.current = false;
      disconnect();
    };
  }, [connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
    };
  }, []);

  return {
    ...state,
    connect,
    disconnect,
    sendMessage,
    reconnectCount: reconnectCount.current,
  };
}

// Hook for subscribing to specific data types
export function useWebSocketSubscription<T = any>(
  messageType: string,
  handler: (data: T) => void
) {
  const { lastMessage } = useWebSocket();

  useEffect(() => {
    if (lastMessage && lastMessage.type === messageType) {
      handler(lastMessage.data);
    }
  }, [lastMessage, messageType, handler]);
}

// Hook for price updates
export function usePriceUpdates(handler: (priceData: any) => void) {
  return useWebSocketSubscription('PRICE_UPDATE', handler);
}

// Hook for position updates
export function usePositionUpdates(handler: (position: any) => void) {
  return useWebSocketSubscription('POSITION_UPDATE', handler);
}

// Hook for order updates
export function useOrderUpdates(handler: (order: any) => void) {
  return useWebSocketSubscription('ORDER_UPDATE', handler);
}

// Hook for alert notifications
export function useAlertUpdates(handler: (alert: any) => void) {
  return useWebSocketSubscription('ALERT', handler);
}
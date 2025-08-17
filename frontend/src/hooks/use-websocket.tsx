import { useEffect, useRef, useState, useCallback, createContext, useContext } from 'react';
import { toast } from 'react-hot-toast';
import { WebSocketMessage, PriceUpdateMessage, PositionUpdateMessage, OrderUpdateMessage, AlertMessage } from '@/types';

interface UseWebSocketOptions {
  url?: string;
  protocols?: string | string[];
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: Event) => void;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  shouldReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface WebSocketState {
  readyState: number;
  lastMessage: WebSocketMessage | null;
  lastError: Event | null;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  reconnectCount: number;
  latency: number | null;
}

const DEFAULT_OPTIONS: UseWebSocketOptions = {
  url: process.env.NODE_ENV === 'development' ? undefined : 'ws://localhost:8000/ws',
  shouldReconnect: false, // 演示模式下禁用重连
  reconnectInterval: 3000,
  maxReconnectAttempts: 3, // 减少重连次数
};

// WebSocket状态常量
const WEBSOCKET_STATES = {
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3,
} as const;

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  const [state, setState] = useState<WebSocketState>({
    readyState: WEBSOCKET_STATES.CLOSED,
    lastMessage: null,
    lastError: null,
    connectionStatus: 'disconnected',
    reconnectCount: 0,
    latency: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastPingTimeRef = useRef<number | null>(null);

  // 清理函数
  const cleanup = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // 发送心跳
  const sendHeartbeat = useCallback(() => {
    if (wsRef.current?.readyState === WEBSOCKET_STATES.OPEN) {
      lastPingTimeRef.current = Date.now();
      wsRef.current.send(JSON.stringify({ type: 'ping', timestamp: lastPingTimeRef.current }));
    }
  }, []);

  // 启动心跳检测
  const startHeartbeat = useCallback(() => {
    heartbeatIntervalRef.current = setInterval(sendHeartbeat, 30000); // 30秒心跳
  }, [sendHeartbeat]);

  // 停止心跳检测
  const stopHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  }, []);

  // 连接WebSocket
  const connect = useCallback(() => {
    try {
      // 在演示模式下完全禁用WebSocket连接
      if (!opts.url || process.env.NODE_ENV === 'development') {
        setState(prev => ({
          ...prev,
          readyState: WEBSOCKET_STATES.CLOSED,
          connectionStatus: 'disconnected',
        }));
        console.log('演示模式：WebSocket已禁用');
        return;
      }

      const ws = new WebSocket(opts.url, opts.protocols);
      wsRef.current = ws;

      setState(prev => ({
        ...prev,
        readyState: WEBSOCKET_STATES.CONNECTING,
        connectionStatus: 'connecting',
      }));

      ws.onopen = (event) => {
        setState(prev => ({
          ...prev,
          readyState: WEBSOCKET_STATES.OPEN,
          connectionStatus: 'connected',
          reconnectCount: 0,
        }));
        
        startHeartbeat();
        opts.onOpen?.(event);
        
        toast.success('WebSocket 连接成功', { duration: 2000 });
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          // 处理心跳响应
          if (message.type === 'pong' && lastPingTimeRef.current) {
            const latency = Date.now() - lastPingTimeRef.current;
            setState(prev => ({ ...prev, latency }));
            return;
          }

          setState(prev => ({ ...prev, lastMessage: message }));
          opts.onMessage?.(message);

          // 处理特定类型的消息
          handleSpecificMessage(message);
          
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket connection error:', error);
        stopHeartbeat();
        opts.onError?.(error);
        
        setState(prev => ({
          ...prev,
          lastError: error,
          connectionStatus: 'error',
        }));
      };

      ws.onclose = (event) => {
        stopHeartbeat();
        opts.onClose?.(event);
        
        setState(prev => ({
          ...prev,
          readyState: WEBSOCKET_STATES.CLOSED,
          connectionStatus: 'disconnected',
        }));
        
        // 只在非正常关闭时重连，且限制重连次数
        if (opts.shouldReconnect && event.code !== 1000 && state.reconnectCount < (opts.maxReconnectAttempts || 3)) {
          const delay = Math.min(1000 * Math.pow(2, state.reconnectCount), 10000); // 指数退避，最大10秒
          
          console.log(`WebSocket将在${delay/1000}秒后重连... (尝试 ${state.reconnectCount + 1}/${opts.maxReconnectAttempts || 3})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            setState(prev => ({ ...prev, reconnectCount: prev.reconnectCount + 1 }));
            connect();
          }, delay);
        }
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setState(prev => ({
        ...prev,
        connectionStatus: 'error',
        lastError: error as Event,
      }));
    }
  }, [opts, state.reconnectCount, startHeartbeat, stopHeartbeat]);

  // 处理特定类型的消息
  const handleSpecificMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'alert':
        handleAlertMessage(message as AlertMessage);
        break;
      case 'price_update':
        // 价格更新通常不需要toast通知
        break;
      case 'position_update':
        handlePositionUpdate(message as PositionUpdateMessage);
        break;
      case 'order_update':
        handleOrderUpdate(message as OrderUpdateMessage);
        break;
      default:
        break;
    }
  }, []);

  // 处理告警消息
  const handleAlertMessage = useCallback((alert: AlertMessage) => {
    const toastOptions = {
      duration: alert.severity === 'critical' ? 0 : 5000, // critical告警不自动消失
      position: 'top-right' as const,
    };

    switch (alert.severity) {
      case 'critical':
        toast.error(`🚨 ${alert.message}`, toastOptions);
        break;
      case 'warning':
        toast(`⚠️ ${alert.message}`, toastOptions);
        break;
      case 'info':
        toast(`ℹ️ ${alert.message}`, toastOptions);
        break;
      default:
        toast(alert.message, toastOptions);
        break;
    }
  }, []);

  // 处理持仓更新
  const handlePositionUpdate = useCallback((update: PositionUpdateMessage) => {
    if (update.data.action === 'opened') {
      toast.success(`📈 新建持仓: ${update.data.symbol} ${update.data.side}`);
    } else if (update.data.action === 'closed') {
      const pnlText = update.data.pnl && update.data.pnl >= 0 ? '盈利' : '亏损';
      toast(`📊 平仓: ${update.data.symbol} (${pnlText}: $${Math.abs(update.data.pnl || 0).toFixed(2)})`);
    }
  }, []);

  // 处理订单更新
  const handleOrderUpdate = useCallback((update: OrderUpdateMessage) => {
    if (update.data.status === 'FILLED') {
      toast.success(`✅ 订单成交: ${update.data.symbol} ${update.data.side}`);
    } else if (update.data.status === 'CANCELED') {
      toast.error(`❌ 订单取消: ${update.data.symbol} ${update.data.side}`);
    }
  }, []);

  // 发送消息
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WEBSOCKET_STATES.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  // 手动重连
  const reconnect = useCallback(() => {
    cleanup();
    setState(prev => ({ ...prev, reconnectCount: 0 }));
    connect();
  }, [cleanup, connect]);

  // 断开连接
  const disconnect = useCallback(() => {
    cleanup();
    setState(prev => ({
      ...prev,
      readyState: WEBSOCKET_STATES.CLOSED,
      connectionStatus: 'disconnected',
    }));
  }, [cleanup]);

  // 初始连接
  useEffect(() => {
    connect();
    return cleanup;
  }, [connect, cleanup]);

  return {
    ...state,
    sendMessage,
    reconnect,
    disconnect,
    isConnected: state.readyState === WEBSOCKET_STATES.OPEN,
    isConnecting: state.readyState === WEBSOCKET_STATES.CONNECTING,
  };
}

// WebSocket Context for global state management
interface WebSocketContextType {
  isConnected: boolean;
  connectionStatus: string;
  latency: number | null;
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: any) => boolean;
  reconnect: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

export function WebSocketProvider({ 
  children, 
  url = 'ws://localhost:8000/ws',
  ...options 
}: { 
  children: React.ReactNode; 
  url?: string; 
} & UseWebSocketOptions) {
  const ws = useWebSocket({ url, ...options });

  const contextValue: WebSocketContextType = {
    isConnected: ws.isConnected,
    connectionStatus: ws.connectionStatus,
    latency: ws.latency,
    lastMessage: ws.lastMessage,
    sendMessage: ws.sendMessage,
    reconnect: ws.reconnect,
  };

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocketContext() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocketContext must be used within a WebSocketProvider');
  }
  return context;
}

// 特定功能的hooks
export function useWebSocketSubscription<T = any>(
  messageType: string,
  onMessage: (data: T) => void
) {
  const { lastMessage } = useWebSocketContext();

  useEffect(() => {
    if (lastMessage?.type === messageType) {
      onMessage(lastMessage.data as T);
    }
  }, [lastMessage, messageType, onMessage]);
}

// 实时价格订阅
export function usePriceUpdates(symbols: string[] = []) {
  const [prices, setPrices] = useState<Record<string, number>>({});
  
  useWebSocketSubscription<PriceUpdateMessage['data']>('price_update', (data) => {
    if (symbols.length === 0 || symbols.includes(data.symbol)) {
      setPrices(prev => ({
        ...prev,
        [data.symbol]: data.price,
      }));
    }
  });

  return prices;
}

// 实时持仓订阅
export function usePositionUpdates() {
  const [positions, setPositions] = useState<any[]>([]);
  
  useWebSocketSubscription<PositionUpdateMessage['data']>('position_update', (data) => {
    setPositions(prev => {
      const existingIndex = prev.findIndex(p => p.id === data.id);
      if (existingIndex >= 0) {
        const updated = [...prev];
        if (data.action === 'closed') {
          updated.splice(existingIndex, 1);
        } else {
          updated[existingIndex] = { ...updated[existingIndex], ...data };
        }
        return updated;
      } else if (data.action === 'opened') {
        return [...prev, data];
      }
      return prev;
    });
  });

  return positions;
}

export default useWebSocket;
import { createContext, useContext, useEffect, useState } from 'react';
import { WebSocketMessage } from '@/types';

interface DemoWebSocketContextType {
  isConnected: boolean;
  connectionStatus: string;
  latency: number | null;
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: any) => boolean;
  reconnect: () => void;
}

const DemoWebSocketContext = createContext<DemoWebSocketContextType | null>(null);

export function DemoWebSocketProvider({ children }: { children: React.ReactNode }) {
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  // 模拟定期接收消息
  useEffect(() => {
    const interval = setInterval(() => {
      // 模拟价格更新消息
      const mockMessage: WebSocketMessage = {
        type: 'price_update',
        data: {
          symbol: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'][Math.floor(Math.random() * 3)],
          price: Math.random() * 100000 + 30000,
          timestamp: new Date().toISOString(),
        },
        timestamp: new Date().toISOString(),
      };
      
      setLastMessage(mockMessage);
    }, 5000); // 每5秒更新一次

    return () => clearInterval(interval);
  }, []);

  const contextValue: DemoWebSocketContextType = {
    isConnected: false, // 演示模式下显示未连接
    connectionStatus: 'disconnected',
    latency: null,
    lastMessage,
    sendMessage: () => {
      console.log('演示模式：消息发送被模拟');
      return true;
    },
    reconnect: () => {
      console.log('演示模式：重连被忽略');
    },
  };

  return (
    <DemoWebSocketContext.Provider value={contextValue}>
      {children}
    </DemoWebSocketContext.Provider>
  );
}

export function useDemoWebSocketContext() {
  const context = useContext(DemoWebSocketContext);
  if (!context) {
    throw new Error('useDemoWebSocketContext must be used within a DemoWebSocketProvider');
  }
  return context;
}

// 对外暴露的接口，在演示模式下使用Demo版本
export function useWebSocketContext() {
  return useDemoWebSocketContext();
}

// 演示模式下的价格订阅
export function usePriceUpdates(symbols: string[] = []) {
  const [prices, setPrices] = useState<Record<string, number>>({
    'BTCUSDT': 67850.00,
    'ETHUSDT': 2685.50,
    'SOLUSDT': 145.60,
    'ADAUSDT': 0.485,
    'XRPUSDT': 0.612,
  });
  
  // 模拟价格变化
  useEffect(() => {
    const interval = setInterval(() => {
      setPrices(prev => {
        const updated = { ...prev };
        Object.keys(updated).forEach(symbol => {
          // 随机变化 ±2%
          const change = (Math.random() - 0.5) * 0.04;
          updated[symbol] = updated[symbol] * (1 + change);
        });
        return updated;
      });
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return prices;
}

// 演示模式下的持仓订阅
export function usePositionUpdates() {
  const [positions] = useState([
    {
      id: 'pos_1',
      symbol: 'BTCUSDT',
      side: 'LONG',
      quantity: 0.1234,
      currentPrice: 68120.00,
      unrealizedPnl: 107.35,
      action: 'active',
    },
    {
      id: 'pos_2', 
      symbol: 'ETHUSDT',
      side: 'LONG',
      quantity: 2.4567,
      currentPrice: 2685.50,
      unrealizedPnl: 111.84,
      action: 'active',
    },
  ]);

  return positions;
}
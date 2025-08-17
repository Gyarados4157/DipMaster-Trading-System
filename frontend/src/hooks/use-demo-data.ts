import { useQuery } from '@tanstack/react-query';
import { PnLData, Position, StrategyStats, Alert, SystemHealth, RiskMetrics } from '@/types';

// 模拟数据生成器
export function useDemoData() {
  // 生成模拟PnL数据
  const generatePnLData = (): PnLData[] => {
    const data: PnLData[] = [];
    const now = new Date();
    let totalPnl = 12000;
    
    for (let i = 100; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 5 * 60 * 1000); // 5分钟间隔
      const change = (Math.random() - 0.5) * 200; // -100 到 +100 的随机变化
      totalPnl += change;
      
      data.push({
        timestamp: timestamp.toISOString(),
        totalPnl: totalPnl,
        realizedPnl: totalPnl * 0.8,
        unrealizedPnl: totalPnl * 0.2,
        balance: 50000 + totalPnl,
        equity: 50000 + totalPnl,
      });
    }
    
    return data;
  };

  // 生成模拟持仓数据
  const generatePositions = (): Position[] => {
    return [
      {
        id: 'pos_1',
        symbol: 'BTCUSDT',
        side: 'LONG',
        quantity: 0.1234,
        entryPrice: 67250.00,
        currentPrice: 68120.00,
        unrealizedPnl: 107.35,
        realizedPnl: 0,
        percentage: 1.29,
        margin: 2000,
        leverage: 3,
        markPrice: 68120.00,
        timestamp: new Date().toISOString(),
        orders: [],
        status: 'ACTIVE',
        maxDrawdown: -0.8,
        holdingTimeMinutes: 45,
        createdAt: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
        updatedAt: new Date().toISOString(),
      },
      {
        id: 'pos_2',
        symbol: 'ETHUSDT',
        side: 'LONG',
        quantity: 2.4567,
        entryPrice: 2640.00,
        currentPrice: 2685.50,
        unrealizedPnl: 111.84,
        realizedPnl: 0,
        percentage: 1.72,
        margin: 1500,
        leverage: 2,
        markPrice: 2685.50,
        timestamp: new Date().toISOString(),
        orders: [],
        status: 'ACTIVE',
        maxDrawdown: -1.2,
        holdingTimeMinutes: 62,
        createdAt: new Date(Date.now() - 62 * 60 * 1000).toISOString(),
        updatedAt: new Date().toISOString(),
      },
      {
        id: 'pos_3',
        symbol: 'SOLUSDT',
        side: 'LONG',
        quantity: 15.2341,
        entryPrice: 142.80,
        currentPrice: 145.60,
        unrealizedPnl: 42.65,
        realizedPnl: 0,
        percentage: 1.96,
        margin: 800,
        leverage: 2.5,
        markPrice: 145.60,
        timestamp: new Date().toISOString(),
        orders: [],
        status: 'ACTIVE',
        maxDrawdown: -0.5,
        holdingTimeMinutes: 28,
        createdAt: new Date(Date.now() - 28 * 60 * 1000).toISOString(),
        updatedAt: new Date().toISOString(),
      },
    ];
  };

  // 生成模拟策略统计
  const generateStrategyStats = (): StrategyStats => {
    return {
      winRate: 82.1,
      avgHoldingTimeMinutes: 96,
      dipBuyingRate: 87.9,
      boundaryComplianceRate: 100,
      maxDrawdown: -2.3,
      totalTrades: 156,
      profitableTrades: 128,
      avgProfit: 25.40,
      avgLoss: -18.20,
      sharpeRatio: 2.15,
      periodStart: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
      periodEnd: new Date().toISOString(),
    };
  };

  // 生成模拟告警
  const generateAlerts = (): Alert[] => {
    return [
      {
        id: 'alert_1',
        type: 'risk_limit',
        severity: 'warning',
        title: '风险限制告警',
        message: 'BTCUSDT持仓接近风险限制 (85%)',
        timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
        acknowledged: false,
        source: 'risk_manager',
        data: { symbol: 'BTCUSDT', riskLevel: 0.85 },
      },
      {
        id: 'alert_2',
        type: 'signal_quality',
        severity: 'info',
        title: '信号质量良好',
        message: 'DipMaster策略信号质量评分: 92/100',
        timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
        acknowledged: true,
        source: 'strategy_monitor',
        data: { qualityScore: 92 },
      },
      {
        id: 'alert_3',
        type: 'execution_delay',
        severity: 'warning',
        title: '执行延迟',
        message: 'ETHUSDT订单执行延迟 +1.2秒',
        timestamp: new Date(Date.now() - 25 * 60 * 1000).toISOString(),
        acknowledged: false,
        source: 'execution_engine',
        data: { symbol: 'ETHUSDT', delayMs: 1200 },
      },
    ];
  };

  // 生成系统健康状态
  const generateSystemHealth = (): SystemHealth => {
    return {
      status: 'healthy',
      uptime: 7 * 24 * 60 * 60 + 12 * 60 * 60 + 34 * 60, // 7天12小时34分钟
      memoryUsage: 72.3,
      cpuUsage: 35.8,
      diskUsage: 45.2,
      activeConnections: 12,
      apiLatency: 45,
      websocketConnected: true,
      lastHeartbeat: new Date().toISOString(),
      components: {
        trading_engine: 'healthy',
        data_provider: 'healthy',
        risk_manager: 'healthy',
        order_executor: 'healthy',
        websocket_server: 'healthy',
      },
    };
  };

  // 生成风险指标
  const generateRiskMetrics = (): RiskMetrics => {
    return {
      var95: 2450.30,
      var99: 3890.75,
      expectedShortfall: 4560.25,
      maxDrawdown: 0.0289,
      sharpeRatio: 2.15,
      volatility: 0.085,
      beta: 0.02,
      correlationWithBTC: 0.65,
      leverage: 2.2,
      marginUsage: 0.68,
      timestamp: new Date().toISOString(),
    };
  };

  return {
    pnlData: generatePnLData(),
    positions: generatePositions(),
    strategyStats: generateStrategyStats(),
    alerts: generateAlerts(),
    systemHealth: generateSystemHealth(),
    riskMetrics: generateRiskMetrics(),
  };
}

// 覆盖原有的API hooks，使用演示数据
export function usePnLData(timeRange: string = '1D') {
  const demoData = useDemoData();
  
  return useQuery({
    queryKey: ['demo-pnl', timeRange],
    queryFn: async () => {
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 500));
      return demoData.pnlData;
    },
    refetchInterval: 30000,
  });
}

export function usePositions() {
  const demoData = useDemoData();
  
  return useQuery({
    queryKey: ['demo-positions'],
    queryFn: async () => {
      await new Promise(resolve => setTimeout(resolve, 300));
      return demoData.positions;
    },
    refetchInterval: 10000,
  });
}

export function useStrategyStats() {
  const demoData = useDemoData();
  
  return useQuery({
    queryKey: ['demo-strategy-stats'],
    queryFn: async () => {
      await new Promise(resolve => setTimeout(resolve, 400));
      return demoData.strategyStats;
    },
    refetchInterval: 30000,
  });
}

export function useAlerts() {
  const demoData = useDemoData();
  
  return useQuery({
    queryKey: ['demo-alerts'],
    queryFn: async () => {
      await new Promise(resolve => setTimeout(resolve, 200));
      return {
        data: demoData.alerts,
        pagination: {
          page: 1,
          limit: 20,
          total: 3,
          totalPages: 1,
          hasNext: false,
          hasPrev: false,
        }
      };
    },
    refetchInterval: 10000,
  });
}

export function useSystemHealth() {
  const demoData = useDemoData();
  
  return useQuery({
    queryKey: ['demo-system-health'],
    queryFn: async () => {
      await new Promise(resolve => setTimeout(resolve, 200));
      return demoData.systemHealth;
    },
    refetchInterval: 30000,
  });
}

export function useRiskMetrics() {
  const demoData = useDemoData();
  
  return useQuery({
    queryKey: ['demo-risk-metrics'],
    queryFn: async () => {
      await new Promise(resolve => setTimeout(resolve, 300));
      return demoData.riskMetrics;
    },
    refetchInterval: 60000,
  });
}
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useAuth } from '@/hooks/use-auth';
import { 
  ApiResponse, 
  PaginatedResponse, 
  Position, 
  Order, 
  PnLData, 
  RiskMetrics, 
  StrategyStats,
  TradingPair,
  Alert,
  SystemHealth,
  ExecutionAnalysis
} from '@/types';

// API 基础配置
const API_BASE_URL = process.env.BACKEND_URL || 'http://localhost:8000';

class ApiClient {
  private baseURL: string;
  private token: string | null = null;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  setToken(token: string | null) {
    this.token = token;
  }

  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    
    const defaultHeaders: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (this.token) {
      defaultHeaders.Authorization = `Bearer ${this.token}`;
    }

    const config: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  async get<T>(endpoint: string): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async put<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }
}

// 创建API客户端实例
const apiClient = new ApiClient(API_BASE_URL);

// 自定义hook来管理API token
export function useApiClient() {
  const { token } = useAuth();
  
  // 更新API客户端的token
  if (token !== apiClient['token']) {
    apiClient.setToken(token);
  }

  return apiClient;
}

// ============ Data Fetching Hooks ============

// 获取PnL数据
export function usePnLData(timeRange: string = '1D') {
  const api = useApiClient();
  
  return useQuery({
    queryKey: ['pnl', timeRange],
    queryFn: async () => {
      const response = await api.get<PnLData[]>(`/api/pnl?range=${timeRange}`);
      return response.data || [];
    },
    refetchInterval: 30000, // 30秒刷新一次
  });
}

// 获取当前持仓
export function usePositions() {
  const api = useApiClient();
  
  return useQuery({
    queryKey: ['positions'],
    queryFn: async () => {
      const response = await api.get<Position[]>('/api/positions/latest');
      return response.data || [];
    },
    refetchInterval: 10000, // 10秒刷新一次
  });
}

// 获取交易历史
export function useTrades(page: number = 1, limit: number = 50) {
  const api = useApiClient();
  
  return useQuery({
    queryKey: ['trades', page, limit],
    queryFn: async () => {
      const response = await api.get<PaginatedResponse<Order>>(`/api/trades?page=${page}&limit=${limit}`);
      return response.data || { data: [], pagination: { page: 1, limit: 50, total: 0, totalPages: 0, hasNext: false, hasPrev: false } };
    },
    refetchInterval: 15000, // 15秒刷新一次
  });
}

// 获取风险指标
export function useRiskMetrics() {
  const api = useApiClient();
  
  return useQuery({
    queryKey: ['risk-metrics'],
    queryFn: async () => {
      const response = await api.get<RiskMetrics>('/api/risk/metrics');
      return response.data;
    },
    refetchInterval: 60000, // 1分钟刷新一次
  });
}

// 获取策略统计
export function useStrategyStats() {
  const api = useApiClient();
  
  return useQuery({
    queryKey: ['strategy-stats'],
    queryFn: async () => {
      const response = await api.get<StrategyStats>('/api/strategy/stats');
      return response.data;
    },
    refetchInterval: 30000, // 30秒刷新一次
  });
}

// 获取市场数据
export function useMarketData() {
  const api = useApiClient();
  
  return useQuery({
    queryKey: ['market-data'],
    queryFn: async () => {
      const response = await api.get<TradingPair[]>('/api/market/data');
      return response.data || [];
    },
    refetchInterval: 5000, // 5秒刷新一次
  });
}

// 获取系统健康状态
export function useSystemHealth() {
  const api = useApiClient();
  
  return useQuery({
    queryKey: ['system-health'],
    queryFn: async () => {
      const response = await api.get<SystemHealth>('/api/system/health');
      return response.data;
    },
    refetchInterval: 30000, // 30秒刷新一次
  });
}

// 获取告警
export function useAlerts(page: number = 1, limit: number = 20) {
  const api = useApiClient();
  
  return useQuery({
    queryKey: ['alerts', page, limit],
    queryFn: async () => {
      const response = await api.get<PaginatedResponse<Alert>>(`/api/alerts?page=${page}&limit=${limit}`);
      return response.data || { data: [], pagination: { page: 1, limit: 20, total: 0, totalPages: 0, hasNext: false, hasPrev: false } };
    },
    refetchInterval: 10000, // 10秒刷新一次
  });
}

// 获取执行分析
export function useExecutionAnalysis(symbol?: string) {
  const api = useApiClient();
  
  return useQuery({
    queryKey: ['execution-analysis', symbol],
    queryFn: async () => {
      const endpoint = symbol ? `/api/execution/analysis/${symbol}` : '/api/execution/analysis';
      const response = await api.get<ExecutionAnalysis[]>(endpoint);
      return response.data || [];
    },
    refetchInterval: 60000, // 1分钟刷新一次
  });
}

// ============ Mutation Hooks ============

// 确认告警
export function useAcknowledgeAlert() {
  const api = useApiClient();
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (alertId: string) => {
      const response = await api.post(`/api/alerts/${alertId}/acknowledge`);
      return response.data;
    },
    onSuccess: () => {
      // 刷新告警列表
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
    },
  });
}

// 手动平仓
export function useClosePosition() {
  const api = useApiClient();
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (positionId: string) => {
      const response = await api.post(`/api/positions/${positionId}/close`);
      return response.data;
    },
    onSuccess: () => {
      // 刷新持仓和交易数据
      queryClient.invalidateQueries({ queryKey: ['positions'] });
      queryClient.invalidateQueries({ queryKey: ['trades'] });
    },
  });
}

// 取消订单
export function useCancelOrder() {
  const api = useApiClient();
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (orderId: string) => {
      const response = await api.post(`/api/orders/${orderId}/cancel`);
      return response.data;
    },
    onSuccess: () => {
      // 刷新订单和持仓数据
      queryClient.invalidateQueries({ queryKey: ['trades'] });
      queryClient.invalidateQueries({ queryKey: ['positions'] });
    },
  });
}

// 更新系统设置
export function useUpdateSettings() {
  const api = useApiClient();
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (settings: any) => {
      const response = await api.put('/api/settings', settings);
      return response.data;
    },
    onSuccess: () => {
      // 根据需要刷新相关数据
      queryClient.invalidateQueries({ queryKey: ['strategy-stats'] });
    },
  });
}

// ============ Mock Data Hooks (for development) ============

// 生成Mock数据的hook，用于开发环境
export function useMockData() {
  // Mock PnL data
  const mockPnLData: PnLData[] = Array.from({ length: 24 }, (_, i) => ({
    timestamp: new Date(Date.now() - (23 - i) * 60 * 60 * 1000).toISOString(),
    totalPnl: Math.random() * 2000 - 1000,
    realizedPnl: Math.random() * 1500 - 750,
    unrealizedPnl: Math.random() * 500 - 250,
    balance: 10000 + Math.random() * 2000 - 1000,
    equity: 10000 + Math.random() * 2000 - 1000,
  }));

  // Mock positions
  const mockPositions: Position[] = [
    {
      id: '1',
      symbol: 'BTCUSDT',
      side: 'LONG',
      quantity: 0.1,
      entryPrice: 43250.00,
      currentPrice: 43650.00,
      unrealizedPnl: 40.00,
      realizedPnl: 0,
      percentage: 0.92,
      margin: 1000,
      leverage: 10,
      markPrice: 43650.00,
      timestamp: new Date().toISOString(),
      orders: [],
      status: 'ACTIVE',
      maxDrawdown: -0.5,
      holdingTimeMinutes: 45,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
  ];

  return {
    pnlData: mockPnLData,
    positions: mockPositions,
  };
}
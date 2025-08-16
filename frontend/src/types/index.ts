// 基础类型定义
export interface BaseEntity {
  id: string;
  createdAt: string;
  updatedAt: string;
}

// 用户相关类型
export interface User {
  id: string;
  username: string;
  email?: string;
  role: 'admin' | 'trader' | 'viewer';
  permissions: string[];
  lastLoginAt?: string;
  isActive: boolean;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

// 交易对和价格数据
export interface TradingPair {
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
  status: 'TRADING' | 'HALT' | 'BREAK';
  price: number;
  priceChange: number;
  priceChangePercent: number;
  volume: number;
  quoteVolume: number;
  high24h: number;
  low24h: number;
  lastUpdate: string;
}

export interface PriceData {
  symbol: string;
  price: number;
  timestamp: string;
  volume?: number;
  change24h?: number;
  changePercent24h?: number;
}

// K线数据
export interface CandlestickData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// 技术指标数据
export interface TechnicalIndicators {
  rsi: number;
  ma20: number;
  ma50: number;
  bollinger: {
    upper: number;
    middle: number;
    lower: number;
  };
  volume_sma: number;
  volatility: number;
}

// 交易信号
export interface TradingSignal {
  id: string;
  symbol: string;
  type: 'ENTRY' | 'EXIT';
  side: 'BUY' | 'SELL';
  price: number;
  confidence: number;
  reason: string;
  indicators: TechnicalIndicators;
  timestamp: string;
  isExecuted: boolean;
}

// 订单相关类型
export interface Order extends BaseEntity {
  symbol: string;
  side: 'BUY' | 'SELL';
  type: 'MARKET' | 'LIMIT' | 'STOP_LOSS' | 'TAKE_PROFIT';
  status: 'NEW' | 'PARTIALLY_FILLED' | 'FILLED' | 'CANCELLED' | 'REJECTED';
  quantity: number;
  price?: number;
  stopPrice?: number;
  executedQuantity: number;
  avgPrice?: number;
  commission?: number;
  commissionAsset?: string;
  fills?: OrderFill[];
  timeInForce: 'GTC' | 'IOC' | 'FOK';
  clientOrderId?: string;
  executionType?: string;
  rejectReason?: string;
}

export interface OrderFill {
  price: number;
  quantity: number;
  commission: number;
  commissionAsset: string;
  timestamp: string;
}

// 持仓相关类型
export interface Position extends BaseEntity {
  symbol: string;
  side: 'LONG' | 'SHORT';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
  percentage: number;
  margin: number;
  leverage: number;
  liquidationPrice?: number;
  markPrice: number;
  timestamp: string;
  orders: Order[];
  status: 'ACTIVE' | 'CLOSED' | 'LIQUIDATED';
  stopLoss?: number;
  takeProfit?: number;
  maxDrawdown: number;
  holdingTimeMinutes: number;
}

// PnL 数据
export interface PnLData {
  timestamp: string;
  totalPnl: number;
  realizedPnl: number;
  unrealizedPnl: number;
  balance: number;
  equity: number;
  margin?: number;
  freeMargin?: number;
  marginLevel?: number;
}

export interface PnLSummary {
  totalPnl: number;
  todayPnl: number;
  weekPnl: number;
  monthPnl: number;
  yearPnl: number;
  allTimePnl: number;
  winRate: number;
  profitFactor: number;
  sharpeRatio: number;
  maxDrawdown: number;
  avgWinningTrade: number;
  avgLosingTrade: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
}

// 风险指标
export interface RiskMetrics {
  var95: number; // Value at Risk 95%
  var99: number; // Value at Risk 99%
  expectedShortfall: number;
  beta: number;
  alpha: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  annualizedVolatility: number;
  leverage: number;
  exposureByExchange: Record<string, number>;
  exposureBySymbol: Record<string, number>;
  correlationRisk: number;
  liquidityRisk: number;
  riskScore: 'LOW' | 'MEDIUM' | 'HIGH';
  lastUpdate: string;
}

// 策略统计
export interface StrategyStats {
  name: string;
  version: string;
  isActive: boolean;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  avgHoldingTimeMinutes: number;
  dipBuyingRate: number;
  boundaryComplianceRate: number;
  profitFactor: number;
  sharpeRatio: number;
  maxDrawdown: number;
  totalPnl: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  consecutiveWins: number;
  consecutiveLosses: number;
  lastUpdate: string;
}

// 执行分析
export interface ExecutionAnalysis {
  symbol: string;
  avgSlippage: number;
  maxSlippage: number;
  fillRate: number;
  avgExecutionTime: number;
  maxExecutionTime: number;
  executionQuality: 'EXCELLENT' | 'GOOD' | 'FAIR' | 'POOR';
  totalOrders: number;
  successfulOrders: number;
  failedOrders: number;
  partialFills: number;
  rejectedOrders: number;
  lastUpdate: string;
}

// 告警类型
export interface Alert extends BaseEntity {
  type: 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  title: string;
  message: string;
  source: string;
  severity: 1 | 2 | 3 | 4 | 5;
  isRead: boolean;
  isAcknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: string;
  relatedSymbol?: string;
  relatedOrderId?: string;
  relatedPositionId?: string;
  metadata?: Record<string, any>;
}

// 系统状态
export interface SystemHealth {
  status: 'HEALTHY' | 'WARNING' | 'CRITICAL' | 'OFFLINE';
  uptime: number;
  lastUpdate: string;
  components: {
    tradingEngine: ComponentStatus;
    dataFeed: ComponentStatus;
    database: ComponentStatus;
    websocket: ComponentStatus;
    riskManager: ComponentStatus;
    orderManager: ComponentStatus;
  };
  metrics: {
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    networkLatency: number;
    activeConnections: number;
    ordersPerSecond: number;
    errorsPerMinute: number;
  };
}

export interface ComponentStatus {
  status: 'ONLINE' | 'OFFLINE' | 'DEGRADED';
  lastCheck: string;
  responseTime: number;
  errorCount: number;
  message?: string;
}

// API 响应类型
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

// WebSocket 消息类型
export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface PriceUpdateMessage extends WebSocketMessage {
  type: 'PRICE_UPDATE';
  data: PriceData;
}

export interface PositionUpdateMessage extends WebSocketMessage {
  type: 'POSITION_UPDATE';
  data: Position;
}

export interface OrderUpdateMessage extends WebSocketMessage {
  type: 'ORDER_UPDATE';
  data: Order;
}

export interface AlertMessage extends WebSocketMessage {
  type: 'ALERT';
  data: Alert;
}

export interface SystemStatusMessage extends WebSocketMessage {
  type: 'SYSTEM_STATUS';
  data: SystemHealth;
}

// 图表数据类型
export interface ChartDataPoint {
  timestamp: string;
  value: number;
  label?: string;
  color?: string;
  metadata?: Record<string, any>;
}

export interface TimeSeriesData {
  name: string;
  data: ChartDataPoint[];
  color?: string;
  type?: 'line' | 'area' | 'bar';
}

// 过滤器和查询参数
export interface TimeRange {
  start: string;
  end: string;
}

export interface TradeFilter {
  symbols?: string[];
  sides?: Array<'BUY' | 'SELL'>;
  statuses?: string[];
  timeRange?: TimeRange;
  minAmount?: number;
  maxAmount?: number;
}

export interface PositionFilter {
  symbols?: string[];
  sides?: Array<'LONG' | 'SHORT'>;
  statuses?: Array<'ACTIVE' | 'CLOSED' | 'LIQUIDATED'>;
  timeRange?: TimeRange;
}

// 配置类型
export interface DashboardConfig {
  refreshInterval: number;
  autoRefresh: boolean;
  defaultTimeRange: string;
  chartsPerRow: number;
  enableSounds: boolean;
  enableNotifications: boolean;
  theme: 'light' | 'dark' | 'system';
  language: string;
  timezone: string;
  currency: 'USD' | 'BTC' | 'ETH';
  precision: {
    price: number;
    quantity: number;
    percentage: number;
  };
}

// 导出数据格式
export interface ExportData {
  type: 'trades' | 'positions' | 'pnl' | 'risk';
  format: 'csv' | 'json' | 'xlsx';
  timeRange: TimeRange;
  filters?: Record<string, any>;
  columns: string[];
}

// 表格列定义
export interface TableColumn<T = any> {
  key: string;
  title: string;
  dataIndex: keyof T;
  width?: number;
  align?: 'left' | 'center' | 'right';
  sorter?: boolean;
  render?: (value: any, record: T, index: number) => React.ReactNode;
  filters?: Array<{ text: string; value: any }>;
  hidden?: boolean;
  fixed?: 'left' | 'right';
}

// 通知设置
export interface NotificationSettings {
  enabled: boolean;
  types: {
    trades: boolean;
    positions: boolean;
    alerts: boolean;
    system: boolean;
    pnl: boolean;
  };
  sounds: {
    enabled: boolean;
    volume: number;
    tradeSound: string;
    alertSound: string;
    errorSound: string;
  };
  desktop: {
    enabled: boolean;
    permission: 'granted' | 'denied' | 'default';
  };
  email: {
    enabled: boolean;
    address?: string;
    frequency: 'immediate' | 'hourly' | 'daily';
  };
}
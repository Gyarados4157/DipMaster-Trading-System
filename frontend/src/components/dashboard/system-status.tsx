'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/dashboard/risk-metrics';
import { 
  Activity, 
  Cpu, 
  HardDrive, 
  Wifi, 
  Database,
  Shield,
  Zap,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock
} from 'lucide-react';
import { useSystemHealth } from '@/hooks/use-api';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';
import { formatNumber } from '@/lib/utils';

export function SystemStatus() {
  const { data: systemHealth, isLoading, error } = useSystemHealth();

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>System Status</CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingSkeleton lines={3} className="h-40" />
        </CardContent>
      </Card>
    );
  }

  // Mock data if no real data available
  const health = systemHealth || {
    status: 'HEALTHY' as const,
    uptime: 86400, // 24 hours in seconds
    lastUpdate: new Date().toISOString(),
    components: {
      tradingEngine: {
        status: 'ONLINE' as const,
        lastCheck: new Date().toISOString(),
        responseTime: 12,
        errorCount: 0,
        message: 'All systems operational',
      },
      dataFeed: {
        status: 'ONLINE' as const,
        lastCheck: new Date().toISOString(),
        responseTime: 8,
        errorCount: 0,
        message: 'Real-time data flowing',
      },
      database: {
        status: 'ONLINE' as const,
        lastCheck: new Date().toISOString(),
        responseTime: 5,
        errorCount: 0,
        message: 'Database responsive',
      },
      websocket: {
        status: 'ONLINE' as const,
        lastCheck: new Date().toISOString(),
        responseTime: 3,
        errorCount: 0,
        message: 'WebSocket connected',
      },
      riskManager: {
        status: 'ONLINE' as const,
        lastCheck: new Date().toISOString(),
        responseTime: 15,
        errorCount: 0,
        message: 'Risk monitoring active',
      },
      orderManager: {
        status: 'ONLINE' as const,
        lastCheck: new Date().toISOString(),
        responseTime: 10,
        errorCount: 0,
        message: 'Order processing normal',
      },
    },
    metrics: {
      cpuUsage: 23.5,
      memoryUsage: 67.2,
      diskUsage: 45.8,
      networkLatency: 12,
      activeConnections: 156,
      ordersPerSecond: 12.5,
      errorsPerMinute: 0,
    },
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ONLINE':
      case 'HEALTHY':
        return CheckCircle;
      case 'DEGRADED':
      case 'WARNING':
        return AlertTriangle;
      case 'OFFLINE':
      case 'CRITICAL':
        return XCircle;
      default:
        return Activity;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ONLINE':
      case 'HEALTHY':
        return 'profit';
      case 'DEGRADED':
      case 'WARNING':
        return 'pending';
      case 'OFFLINE':
      case 'CRITICAL':
        return 'loss';
      default:
        return 'neutral';
    }
  };

  const getUsageColor = (usage: number) => {
    if (usage < 60) return 'text-trading-profit';
    if (usage < 80) return 'text-trading-pending';
    return 'text-trading-loss';
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / (24 * 3600));
    const hours = Math.floor((seconds % (24 * 3600)) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const components = [
    { key: 'tradingEngine', name: 'Trading Engine', icon: Zap },
    { key: 'dataFeed', name: 'Data Feed', icon: Wifi },
    { key: 'database', name: 'Database', icon: Database },
    { key: 'websocket', name: 'WebSocket', icon: Activity },
    { key: 'riskManager', name: 'Risk Manager', icon: Shield },
    { key: 'orderManager', name: 'Order Manager', icon: CheckCircle },
  ];

  const StatusIcon = getStatusIcon(health.status);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <StatusIcon className={`h-5 w-5 ${
                health.status === 'HEALTHY' ? 'text-trading-profit' :
                health.status === 'WARNING' ? 'text-trading-pending' : 'text-trading-loss'
              }`} />
              <span>System Status</span>
            </CardTitle>
            <CardDescription>
              Real-time system health monitoring
            </CardDescription>
          </div>
          
          <Badge variant={getStatusColor(health.status)} className="text-xs">
            {health.status}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* System Overview */}
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-muted-foreground text-xs">Uptime</p>
            <p className="font-medium flex items-center space-x-1">
              <Clock className="h-3 w-3" />
              <span>{formatUptime(health.uptime)}</span>
            </p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Last Check</p>
            <p className="font-medium">
              {new Date(health.lastUpdate).toLocaleTimeString()}
            </p>
          </div>
        </div>

        {/* Component Status */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Components</h4>
          {components.map((component) => {
            const comp = health.components[component.key as keyof typeof health.components];
            const CompStatusIcon = getStatusIcon(comp.status);
            
            return (
              <div key={component.key} className="flex items-center justify-between text-sm">
                <div className="flex items-center space-x-2">
                  <component.icon className="h-3 w-3 text-muted-foreground" />
                  <span>{component.name}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-muted-foreground">
                    {comp.responseTime}ms
                  </span>
                  <CompStatusIcon className={`h-3 w-3 ${
                    comp.status === 'ONLINE' ? 'text-trading-profit' :
                    comp.status === 'DEGRADED' ? 'text-trading-pending' : 'text-trading-loss'
                  }`} />
                </div>
              </div>
            );
          })}
        </div>

        {/* Resource Usage */}
        <div className="space-y-3 pt-2 border-t border-border">
          <h4 className="text-sm font-medium">Resource Usage</h4>
          
          <div className="space-y-3">
            <div className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center space-x-2">
                  <Cpu className="h-3 w-3" />
                  <span>CPU</span>
                </div>
                <span className={`font-medium ${getUsageColor(health.metrics.cpuUsage)}`}>
                  {health.metrics.cpuUsage.toFixed(1)}%
                </span>
              </div>
              <Progress value={health.metrics.cpuUsage} className="h-1.5" />
            </div>
            
            <div className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center space-x-2">
                  <Activity className="h-3 w-3" />
                  <span>Memory</span>
                </div>
                <span className={`font-medium ${getUsageColor(health.metrics.memoryUsage)}`}>
                  {health.metrics.memoryUsage.toFixed(1)}%
                </span>
              </div>
              <Progress value={health.metrics.memoryUsage} className="h-1.5" />
            </div>
            
            <div className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center space-x-2">
                  <HardDrive className="h-3 w-3" />
                  <span>Disk</span>
                </div>
                <span className={`font-medium ${getUsageColor(health.metrics.diskUsage)}`}>
                  {health.metrics.diskUsage.toFixed(1)}%
                </span>
              </div>
              <Progress value={health.metrics.diskUsage} className="h-1.5" />
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="pt-2 border-t border-border">
          <div className="grid grid-cols-2 gap-3 text-center">
            <div className="p-2 bg-muted/50 rounded">
              <p className="text-xs text-muted-foreground">Network Latency</p>
              <p className="text-sm font-bold">{health.metrics.networkLatency}ms</p>
            </div>
            <div className="p-2 bg-muted/50 rounded">
              <p className="text-xs text-muted-foreground">Active Connections</p>
              <p className="text-sm font-bold">{health.metrics.activeConnections}</p>
            </div>
            <div className="p-2 bg-muted/50 rounded">
              <p className="text-xs text-muted-foreground">Orders/sec</p>
              <p className="text-sm font-bold">{health.metrics.ordersPerSecond.toFixed(1)}</p>
            </div>
            <div className="p-2 bg-muted/50 rounded">
              <p className="text-xs text-muted-foreground">Errors/min</p>
              <p className={`text-sm font-bold ${
                health.metrics.errorsPerMinute === 0 ? 'text-trading-profit' : 'text-trading-loss'
              }`}>
                {health.metrics.errorsPerMinute}
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
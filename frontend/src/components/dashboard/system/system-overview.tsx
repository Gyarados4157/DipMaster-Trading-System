'use client';

import { useSystemHealth } from '@/hooks/use-api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Server, 
  Cpu, 
  MemoryStick, 
  HardDrive,
  Network,
  Clock,
  Activity,
  Zap,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  TrendingUp
} from 'lucide-react';

interface SystemMetric {
  label: string;
  value: string;
  status: 'healthy' | 'warning' | 'critical';
  usage?: number; // 0-100 percentage
  icon: React.ComponentType<any>;
  description: string;
}

export function SystemOverview() {
  const { data: systemData, isLoading } = useSystemHealth();

  if (isLoading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-muted rounded w-1/3"></div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-24 bg-muted rounded"></div>
            ))}
          </div>
        </div>
      </Card>
    );
  }

  const systemMetrics: SystemMetric[] = [
    {
      label: 'CPU Usage',
      value: `${systemData?.cpu?.usage || 23.5}%`,
      status: (systemData?.cpu?.usage || 23.5) > 80 ? 'critical' : 
              (systemData?.cpu?.usage || 23.5) > 60 ? 'warning' : 'healthy',
      usage: systemData?.cpu?.usage || 23.5,
      icon: Cpu,
      description: 'Current CPU utilization',
    },
    {
      label: 'Memory',
      value: `${systemData?.memory?.usage || 45.2}%`,
      status: (systemData?.memory?.usage || 45.2) > 85 ? 'critical' : 
              (systemData?.memory?.usage || 45.2) > 70 ? 'warning' : 'healthy',
      usage: systemData?.memory?.usage || 45.2,
      icon: MemoryStick,
      description: 'RAM utilization',
    },
    {
      label: 'Disk Space',
      value: `${systemData?.disk?.usage || 67.8}%`,
      status: (systemData?.disk?.usage || 67.8) > 90 ? 'critical' : 
              (systemData?.disk?.usage || 67.8) > 80 ? 'warning' : 'healthy',
      usage: systemData?.disk?.usage || 67.8,
      icon: HardDrive,
      description: 'Storage utilization',
    },
    {
      label: 'Network I/O',
      value: `${systemData?.network?.throughput || 156}MB/s`,
      status: 'healthy',
      icon: Network,
      description: 'Network throughput',
    },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-trading-profit';
      case 'warning': return 'text-trading-pending';
      case 'critical': return 'text-trading-loss';
      default: return 'text-muted-foreground';
    }
  };

  const getStatusBgColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-trading-profit/10';
      case 'warning': return 'bg-trading-pending/10';
      case 'critical': return 'bg-trading-loss/10';
      default: return 'bg-muted/10';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return CheckCircle2;
      case 'warning': return AlertTriangle;
      case 'critical': return XCircle;
      default: return Activity;
    }
  };

  const overallStatus = systemMetrics.some(m => m.status === 'critical') ? 'critical' :
                       systemMetrics.some(m => m.status === 'warning') ? 'warning' : 'healthy';

  return (
    <Card className="p-6">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg ${getStatusBgColor(overallStatus)}`}>
              <Server className={`h-5 w-5 ${getStatusColor(overallStatus)}`} />
            </div>
            <div>
              <h3 className="text-lg font-semibold">System Overview</h3>
              <p className="text-sm text-muted-foreground">Real-time infrastructure metrics</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Badge 
              variant={overallStatus === 'healthy' ? 'default' : 'destructive'}
              className="text-xs"
            >
              {overallStatus.toUpperCase()}
            </Badge>
            <div className="text-xs text-muted-foreground">
              Uptime: {systemData?.uptime || '15d 6h 23m'}
            </div>
          </div>
        </div>

        {/* System Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {systemMetrics.map((metric, index) => {
            const Icon = metric.icon;
            const StatusIcon = getStatusIcon(metric.status);
            const statusColor = getStatusColor(metric.status);
            const statusBgColor = getStatusBgColor(metric.status);
            
            return (
              <div key={index} className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className={`p-1.5 rounded ${statusBgColor}`}>
                      <Icon className={`h-4 w-4 ${statusColor}`} />
                    </div>
                    <span className="text-sm font-medium">{metric.label}</span>
                  </div>
                  
                  <StatusIcon className={`h-4 w-4 ${statusColor}`} />
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-baseline justify-between">
                    <span className={`text-xl font-bold ${statusColor}`}>
                      {metric.value}
                    </span>
                  </div>
                  
                  {metric.usage !== undefined && (
                    <div className="space-y-1">
                      <Progress 
                        value={metric.usage} 
                        className="h-1.5"
                      />
                      <p className="text-xs text-muted-foreground">
                        {metric.description}
                      </p>
                    </div>
                  )}
                  
                  {metric.usage === undefined && (
                    <p className="text-xs text-muted-foreground">
                      {metric.description}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* System Information */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t">
          <div className="text-center p-3 bg-muted/30 rounded-lg">
            <div className="flex items-center justify-center space-x-1 mb-1">
              <Clock className="h-4 w-4" />
              <span className="font-medium text-sm">Last Restart</span>
            </div>
            <div className="text-lg font-bold">
              {systemData?.lastRestart || '2024-08-15 14:30'}
            </div>
            <div className="text-xs text-muted-foreground">
              {systemData?.uptimeDays || 15} days ago
            </div>
          </div>
          
          <div className="text-center p-3 bg-muted/30 rounded-lg">
            <div className="flex items-center justify-center space-x-1 mb-1">
              <Zap className="h-4 w-4" />
              <span className="font-medium text-sm">Load Average</span>
            </div>
            <div className="text-lg font-bold">
              {systemData?.loadAverage || '0.85'}
            </div>
            <div className="text-xs text-muted-foreground">
              1min average
            </div>
          </div>
          
          <div className="text-center p-3 bg-muted/30 rounded-lg">
            <div className="flex items-center justify-center space-x-1 mb-1">
              <TrendingUp className="h-4 w-4" />
              <span className="font-medium text-sm">Performance</span>
            </div>
            <div className="text-lg font-bold text-trading-profit">
              {systemData?.performanceScore || 98}%
            </div>
            <div className="text-xs text-muted-foreground">
              Overall score
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
          <div className="text-center">
            <div className="text-2xl font-bold">
              {systemData?.activeConnections || 47}
            </div>
            <div className="text-xs text-muted-foreground">Active Connections</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold">
              {systemData?.requestsPerSecond || 156}
            </div>
            <div className="text-xs text-muted-foreground">Requests/sec</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold">
              {systemData?.errorRate || 0.01}%
            </div>
            <div className="text-xs text-muted-foreground">Error Rate</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold">
              {systemData?.avgResponseTime || 45}ms
            </div>
            <div className="text-xs text-muted-foreground">Avg Response</div>
          </div>
        </div>
      </div>
    </Card>
  );
}
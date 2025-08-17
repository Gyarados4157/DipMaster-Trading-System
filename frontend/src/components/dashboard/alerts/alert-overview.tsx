'use client';

import { useAlertOverviewData } from '@/hooks/use-api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, AlertTriangle, Info, CheckCircle2, Clock } from 'lucide-react';

interface AlertMetric {
  type: 'critical' | 'warning' | 'info' | 'resolved';
  count: number;
  change: number;
  icon: React.ComponentType<any>;
  color: string;
  bgColor: string;
}

export function AlertOverview() {
  const { data: overviewData, isLoading } = useAlertOverviewData();

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i} className="p-6 animate-pulse">
            <div className="h-4 bg-muted rounded w-3/4 mb-2"></div>
            <div className="h-8 bg-muted rounded w-1/2 mb-2"></div>
            <div className="h-3 bg-muted rounded w-1/3"></div>
          </Card>
        ))}
      </div>
    );
  }

  const alertMetrics: AlertMetric[] = [
    {
      type: 'critical',
      count: overviewData?.critical || 1,
      change: overviewData?.criticalChange || 0,
      icon: AlertCircle,
      color: 'text-trading-loss',
      bgColor: 'bg-trading-loss/10',
    },
    {
      type: 'warning',
      count: overviewData?.warning || 2,
      change: overviewData?.warningChange || +1,
      icon: AlertTriangle,
      color: 'text-trading-pending',
      bgColor: 'bg-trading-pending/10',
    },
    {
      type: 'info',
      count: overviewData?.info || 5,
      change: overviewData?.infoChange || +3,
      icon: Info,
      color: 'text-dipmaster-blue',
      bgColor: 'bg-dipmaster-blue/10',
    },
    {
      type: 'resolved',
      count: overviewData?.resolved || 47,
      change: overviewData?.resolvedChange || +12,
      icon: CheckCircle2,
      color: 'text-trading-profit',
      bgColor: 'bg-trading-profit/10',
    },
  ];

  const formatTypeLabel = (type: string) => {
    switch (type) {
      case 'critical': return 'Critical';
      case 'warning': return 'Warning';
      case 'info': return 'Info';
      case 'resolved': return 'Resolved';
      default: return type;
    }
  };

  return (
    <div className="space-y-4">
      {/* Alert Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {alertMetrics.map((metric, index) => {
          const Icon = metric.icon;
          const changeText = metric.change > 0 ? `+${metric.change}` : metric.change.toString();
          const changeColor = metric.change > 0 ? 'text-trading-loss' : 'text-trading-profit';
          
          return (
            <Card key={index} className="p-6 relative overflow-hidden">
              <div className="flex items-center justify-between mb-4">
                <div className={`p-2 rounded-lg ${metric.bgColor}`}>
                  <Icon className={`h-5 w-5 ${metric.color}`} />
                </div>
                <Badge 
                  variant={metric.type === 'critical' ? 'destructive' : 
                           metric.type === 'warning' ? 'secondary' : 
                           metric.type === 'resolved' ? 'default' : 'outline'}
                  className="text-xs"
                >
                  {formatTypeLabel(metric.type).toUpperCase()}
                </Badge>
              </div>
              
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">{formatTypeLabel(metric.type)} Alerts</p>
                <p className="text-3xl font-bold">{metric.count}</p>
                
                <div className="flex items-center space-x-2">
                  <span className={`text-sm font-medium ${changeColor}`}>
                    {changeText}
                  </span>
                  <span className="text-xs text-muted-foreground">last 24h</span>
                </div>
              </div>

              {/* Background decoration */}
              <div className={`absolute -top-4 -right-4 w-16 h-16 rounded-full opacity-5 ${metric.color.replace('text-', 'bg-')}`} />
            </Card>
          );
        })}
      </div>

      {/* Recent Critical Alerts */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-trading-loss/10">
              <AlertCircle className="h-4 w-4 text-trading-loss" />
            </div>
            <div>
              <h3 className="font-semibold">Recent Critical Alerts</h3>
              <p className="text-sm text-muted-foreground">Last 24 hours</p>
            </div>
          </div>
          <Badge variant="destructive" className="text-xs">
            URGENT
          </Badge>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-trading-loss/5 rounded-lg border border-trading-loss/20">
            <div className="flex items-center space-x-3">
              <AlertCircle className="h-4 w-4 text-trading-loss" />
              <div>
                <p className="font-medium">Portfolio drawdown exceeded 3%</p>
                <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  <span>2 minutes ago</span>
                  <Badge variant="outline" className="text-xs">RISK</Badge>
                </div>
              </div>
            </div>
            <Badge variant="destructive" className="text-xs">
              ACTIVE
            </Badge>
          </div>

          {overviewData?.recentCritical && overviewData.recentCritical.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              <CheckCircle2 className="h-12 w-12 mx-auto mb-2 text-trading-profit" />
              <p className="font-medium">No critical alerts</p>
              <p className="text-sm">All systems operating normally</p>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
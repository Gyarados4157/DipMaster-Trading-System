'use client';

import { useState } from 'react';
import { useAlertHistoryData } from '@/hooks/use-api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  AlertCircle, 
  AlertTriangle, 
  Info, 
  CheckCircle2, 
  Clock, 
  Eye,
  EyeOff,
  Download,
  RefreshCw,
  User
} from 'lucide-react';

interface Alert {
  id: string;
  type: 'critical' | 'warning' | 'info' | 'resolved';
  title: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  acknowledgedBy?: string;
  category: string;
  source: string;
  details?: any;
}

export function AlertHistory() {
  const [currentPage, setCurrentPage] = useState(1);
  const [showAcknowledged, setShowAcknowledged] = useState(true);
  const { data: alertsData, isLoading, refetch } = useAlertHistoryData(currentPage, showAcknowledged);

  if (isLoading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-muted rounded w-1/3"></div>
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="space-y-2">
              <div className="h-4 bg-muted rounded w-3/4"></div>
              <div className="h-3 bg-muted rounded w-1/2"></div>
            </div>
          ))}
        </div>
      </Card>
    );
  }

  // Mock alert data
  const mockAlerts: Alert[] = alertsData?.alerts || [
    {
      id: '1',
      type: 'critical',
      title: 'Portfolio Drawdown Alert',
      message: 'Portfolio drawdown has exceeded the 3% threshold',
      timestamp: new Date(Date.now() - 2 * 60 * 1000),
      acknowledged: false,
      category: 'Risk Management',
      source: 'Risk Monitor',
    },
    {
      id: '2',
      type: 'warning',
      title: 'High Volatility Detected',
      message: 'BTC volatility spike detected - consider position sizing adjustments',
      timestamp: new Date(Date.now() - 15 * 60 * 1000),
      acknowledged: false,
      category: 'Market Analysis',
      source: 'Signal Detector',
    },
    {
      id: '3',
      type: 'warning',
      title: 'Exchange Connectivity Issue',
      message: 'Temporary connection issues with Binance WebSocket',
      timestamp: new Date(Date.now() - 30 * 60 * 1000),
      acknowledged: true,
      acknowledgedBy: 'System Admin',
      category: 'System',
      source: 'WebSocket Monitor',
    },
    {
      id: '4',
      type: 'info',
      title: 'Strategy Performance Update',
      message: 'DipMaster strategy achieved 85% win rate in the last 24 hours',
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
      acknowledged: true,
      category: 'Performance',
      source: 'Strategy Monitor',
    },
    {
      id: '5',
      type: 'resolved',
      title: 'Position Size Limit Restored',
      message: 'Position sizing has returned to normal parameters',
      timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000),
      acknowledged: true,
      category: 'Risk Management',
      source: 'Position Manager',
    },
  ];

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'critical': return AlertCircle;
      case 'warning': return AlertTriangle;
      case 'info': return Info;
      case 'resolved': return CheckCircle2;
      default: return Info;
    }
  };

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'critical': return 'text-trading-loss';
      case 'warning': return 'text-trading-pending';
      case 'info': return 'text-dipmaster-blue';
      case 'resolved': return 'text-trading-profit';
      default: return 'text-muted-foreground';
    }
  };

  const getBadgeVariant = (type: string) => {
    switch (type) {
      case 'critical': return 'destructive';
      case 'warning': return 'secondary';
      case 'info': return 'outline';
      case 'resolved': return 'default';
      default: return 'outline';
    }
  };

  const formatTimeAgo = (timestamp: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - timestamp.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  const handleAcknowledge = (alertId: string) => {
    // Handle alert acknowledgment
    console.log('Acknowledging alert:', alertId);
  };

  const handleExportAlerts = () => {
    // Handle CSV export
    console.log('Exporting alerts to CSV');
  };

  const filteredAlerts = showAcknowledged 
    ? mockAlerts 
    : mockAlerts.filter(alert => !alert.acknowledged);

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold">Alert History</h3>
          <p className="text-sm text-muted-foreground">
            {filteredAlerts.length} alerts total
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowAcknowledged(!showAcknowledged)}
            className="text-xs"
          >
            {showAcknowledged ? <EyeOff className="h-3 w-3 mr-1" /> : <Eye className="h-3 w-3 mr-1" />}
            {showAcknowledged ? 'Hide' : 'Show'} Acknowledged
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            className="text-xs"
          >
            <RefreshCw className="h-3 w-3 mr-1" />
            Refresh
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={handleExportAlerts}
            className="text-xs"
          >
            <Download className="h-3 w-3 mr-1" />
            Export
          </Button>
        </div>
      </div>

      {/* Alert List */}
      <div className="space-y-3">
        {filteredAlerts.map((alert) => {
          const Icon = getAlertIcon(alert.type);
          const iconColor = getAlertColor(alert.type);
          
          return (
            <div 
              key={alert.id} 
              className={`
                p-4 rounded-lg border transition-all hover:shadow-sm
                ${alert.acknowledged ? 'bg-muted/30 opacity-75' : 'bg-background'}
                ${alert.type === 'critical' && !alert.acknowledged ? 'border-trading-loss/30 bg-trading-loss/5' : ''}
              `}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3 flex-1">
                  <div className={`p-1 rounded ${alert.acknowledged ? 'opacity-50' : ''}`}>
                    <Icon className={`h-4 w-4 ${iconColor}`} />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2 mb-1">
                      <h4 className={`font-medium ${alert.acknowledged ? 'opacity-75' : ''}`}>
                        {alert.title}
                      </h4>
                      <Badge variant={getBadgeVariant(alert.type)} className="text-xs">
                        {alert.type.toUpperCase()}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {alert.category}
                      </Badge>
                    </div>
                    
                    <p className={`text-sm text-muted-foreground mb-2 ${alert.acknowledged ? 'opacity-75' : ''}`}>
                      {alert.message}
                    </p>
                    
                    <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                      <div className="flex items-center space-x-1">
                        <Clock className="h-3 w-3" />
                        <span>{formatTimeAgo(alert.timestamp)}</span>
                      </div>
                      
                      <div className="flex items-center space-x-1">
                        <span>Source: {alert.source}</span>
                      </div>
                      
                      {alert.acknowledged && alert.acknowledgedBy && (
                        <div className="flex items-center space-x-1">
                          <User className="h-3 w-3" />
                          <span>Acked by {alert.acknowledgedBy}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2 ml-4">
                  {!alert.acknowledged && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleAcknowledge(alert.id)}
                      className="text-xs"
                    >
                      Acknowledge
                    </Button>
                  )}
                  
                  {alert.acknowledged && (
                    <Badge variant="outline" className="text-xs">
                      <CheckCircle2 className="h-3 w-3 mr-1" />
                      Acknowledged
                    </Badge>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between pt-4 mt-4 border-t">
        <p className="text-xs text-muted-foreground">
          Showing {filteredAlerts.length} of {alertsData?.total || filteredAlerts.length} alerts
        </p>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            disabled={currentPage === 1}
            onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
            className="text-xs"
          >
            Previous
          </Button>
          
          <span className="text-xs text-muted-foreground">
            Page {currentPage}
          </span>
          
          <Button
            variant="outline"
            size="sm"
            disabled={filteredAlerts.length < 20}
            onClick={() => setCurrentPage(prev => prev + 1)}
            className="text-xs"
          >
            Next
          </Button>
        </div>
      </div>
    </Card>
  );
}
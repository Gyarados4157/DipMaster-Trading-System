'use client';

import { useState } from 'react';
import { useWebSocketContext } from '@/hooks/use-demo-websocket';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Wifi, 
  WifiOff, 
  Signal, 
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Activity,
  Clock,
  Zap
} from 'lucide-react';

interface WebSocketStatusProps {
  className?: string;
  compact?: boolean;
}

export function WebSocketStatus({ className, compact = false }: WebSocketStatusProps) {
  const [showDetails, setShowDetails] = useState(false);
  const {
    connectionStatus,
    isConnected,
    reconnectCount,
    error,
    latencyMs,
    messagesReceived,
    connect,
    disconnect
  } = useWebSocketContext();

  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <CheckCircle2 className="h-4 w-4 text-trading-profit" />;
      case 'connecting':
        return <RefreshCw className="h-4 w-4 text-trading-pending animate-spin" />;
      case 'disconnected':
        return <AlertCircle className="h-4 w-4 text-trading-loss" />;
      default:
        return <WifiOff className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'text-trading-profit';
      case 'connecting':
        return 'text-trading-pending';
      case 'disconnected':
        return 'text-trading-loss';
      default:
        return 'text-muted-foreground';
    }
  };

  const getStatusBadgeVariant = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'default' as const;
      case 'connecting':
        return 'secondary' as const;
      case 'disconnected':
        return 'destructive' as const;
      default:
        return 'outline' as const;
    }
  };

  const getLatencyStatus = () => {
    if (latencyMs < 100) return { color: 'text-trading-profit', status: 'Excellent' };
    if (latencyMs < 250) return { color: 'text-dipmaster-green', status: 'Good' };
    if (latencyMs < 500) return { color: 'text-trading-pending', status: 'Fair' };
    return { color: 'text-trading-loss', status: 'Poor' };
  };

  const latencyStatus = getLatencyStatus();

  if (compact) {
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        {getStatusIcon()}
        <span className={`text-sm font-medium ${getStatusColor()}`}>
          {connectionStatus.charAt(0).toUpperCase() + connectionStatus.slice(1)}
        </span>
        {isConnected && (
          <Badge variant="outline" className="text-xs">
            {latencyMs}ms
          </Badge>
        )}
      </div>
    );
  }

  return (
    <Card className={`p-4 ${className}`}>
      <div className="space-y-4">
        {/* Main Status */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {getStatusIcon()}
            <div>
              <h3 className="font-medium">WebSocket Connection</h3>
              <p className={`text-sm ${getStatusColor()}`}>
                {connectionStatus.charAt(0).toUpperCase() + connectionStatus.slice(1)}
                {error && ` - ${error}`}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Badge variant={getStatusBadgeVariant()} className="text-xs">
              {connectionStatus.toUpperCase()}
            </Badge>
            
            {!isConnected && (
              <Button
                variant="outline"
                size="sm"
                onClick={connect}
                className="text-xs"
              >
                <RefreshCw className="h-3 w-3 mr-1" />
                Reconnect
              </Button>
            )}
          </div>
        </div>

        {/* Connection Metrics */}
        {isConnected && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t">
            <div className="text-center">
              <div className="flex items-center justify-center space-x-1 mb-1">
                <Signal className="h-4 w-4" />
                <span className="text-sm font-medium">Latency</span>
              </div>
              <div className={`text-lg font-bold ${latencyStatus.color}`}>
                {latencyMs}ms
              </div>
              <div className="text-xs text-muted-foreground">
                {latencyStatus.status}
              </div>
            </div>
            
            <div className="text-center">
              <div className="flex items-center justify-center space-x-1 mb-1">
                <Activity className="h-4 w-4" />
                <span className="text-sm font-medium">Messages</span>
              </div>
              <div className="text-lg font-bold">
                {messagesReceived.toLocaleString()}
              </div>
              <div className="text-xs text-muted-foreground">
                Received
              </div>
            </div>
            
            <div className="text-center">
              <div className="flex items-center justify-center space-x-1 mb-1">
                <Zap className="h-4 w-4" />
                <span className="text-sm font-medium">Reconnects</span>
              </div>
              <div className="text-lg font-bold">
                {reconnectCount}
              </div>
              <div className="text-xs text-muted-foreground">
                Attempts
              </div>
            </div>
          </div>
        )}

        {/* Connection Details */}
        <div className="pt-4 border-t">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowDetails(!showDetails)}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            {showDetails ? 'Hide' : 'Show'} Details
          </Button>
          
          {showDetails && (
            <div className="mt-3 space-y-2 text-xs text-muted-foreground">
              <div className="flex justify-between">
                <span>Endpoint:</span>
                <span className="font-mono">
                  {process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span>Protocol:</span>
                <span>WebSocket</span>
              </div>
              
              <div className="flex justify-between">
                <span>Auto-reconnect:</span>
                <span>Enabled</span>
              </div>
              
              <div className="flex justify-between">
                <span>Heartbeat:</span>
                <span>30s</span>
              </div>
              
              {isConnected && (
                <div className="flex justify-between">
                  <span>Session duration:</span>
                  <span>
                    <Clock className="h-3 w-3 inline mr-1" />
                    Active
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between pt-4 border-t">
          <div className="text-xs text-muted-foreground">
            Real-time data feed status
          </div>
          
          <div className="flex items-center space-x-2">
            {isConnected && (
              <Button
                variant="outline"
                size="sm"
                onClick={disconnect}
                className="text-xs"
              >
                Disconnect
              </Button>
            )}
            
            <div className="flex items-center space-x-1">
              <div className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-trading-profit animate-pulse' : 'bg-trading-loss'
              }`} />
              <span className="text-xs text-muted-foreground">
                {isConnected ? 'Live' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
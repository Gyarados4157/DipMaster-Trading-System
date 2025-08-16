'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  TrendingUp, 
  TrendingDown, 
  Clock, 
  X,
  ExternalLink,
  AlertTriangle
} from 'lucide-react';
import { usePositions, useClosePosition } from '@/hooks/use-api';
import { formatCurrency, formatPercentage, getPnLColorClass, formatTime } from '@/lib/utils';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';
import { toast } from 'react-hot-toast';

export function ActivePositions() {
  const { data: positions, isLoading, error } = usePositions();
  const closePositionMutation = useClosePosition();

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Active Positions</CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingSkeleton lines={3} className="h-60" />
        </CardContent>
      </Card>
    );
  }

  const activePositions = positions?.filter(pos => pos.status === 'ACTIVE') || [];

  const handleClosePosition = async (positionId: string, symbol: string) => {
    try {
      await closePositionMutation.mutateAsync(positionId);
      toast.success(`Position ${symbol} closed successfully`);
    } catch (error) {
      toast.error(`Failed to close position ${symbol}`);
    }
  };

  const getPositionRiskLevel = (position: any) => {
    const holdingTime = position.holdingTimeMinutes || 0;
    const unrealizedPnl = position.unrealizedPnl || 0;
    const percentage = Math.abs(unrealizedPnl / (position.quantity * position.entryPrice)) * 100;

    if (holdingTime > 150 || percentage > 3) return 'high';
    if (holdingTime > 90 || percentage > 1.5) return 'medium';
    return 'low';
  };

  const getRiskBadgeVariant = (level: string) => {
    switch (level) {
      case 'high': return 'loss';
      case 'medium': return 'pending';
      case 'low': return 'profit';
      default: return 'neutral';
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5" />
              <span>Active Positions</span>
            </CardTitle>
            <CardDescription>
              {activePositions.length} open position{activePositions.length !== 1 ? 's' : ''}
            </CardDescription>
          </div>
          
          {activePositions.length > 0 && (
            <Badge variant="profit" className="text-xs">
              Live
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {activePositions.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-sm">No active positions</p>
            <p className="text-xs mt-1">Strategy is monitoring for entry signals</p>
          </div>
        ) : (
          activePositions.map((position) => {
            const riskLevel = getPositionRiskLevel(position);
            const holdingMinutes = position.holdingTimeMinutes || 0;
            const timeToExit = Math.max(0, 180 - holdingMinutes); // Max 180min holding
            
            return (
              <div
                key={position.id}
                className="border border-border rounded-lg p-4 space-y-3 bg-card/50 hover:bg-card transition-colors"
              >
                {/* Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-sm">{position.symbol}</span>
                    <Badge 
                      variant={position.side === 'LONG' ? 'profit' : 'loss'}
                      className="text-xs"
                    >
                      {position.side}
                    </Badge>
                    <Badge 
                      variant={getRiskBadgeVariant(riskLevel)}
                      className="text-xs capitalize"
                    >
                      {riskLevel} risk
                    </Badge>
                  </div>
                  
                  <div className="flex items-center space-x-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6"
                      onClick={() => handleClosePosition(position.id, position.symbol)}
                      disabled={closePositionMutation.isPending}
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                </div>

                {/* Position Details */}
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-muted-foreground text-xs">Quantity</p>
                    <p className="font-medium">{position.quantity}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-xs">Entry Price</p>
                    <p className="font-medium">{formatCurrency(position.entryPrice)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-xs">Current Price</p>
                    <p className="font-medium">{formatCurrency(position.currentPrice)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-xs">Unrealized P&L</p>
                    <p className={`font-bold ${getPnLColorClass(position.unrealizedPnl)}`}>
                      {formatCurrency(position.unrealizedPnl)}
                    </p>
                  </div>
                </div>

                {/* Progress and Timing */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center space-x-1">
                      <Clock className="h-3 w-3" />
                      <span>Holding: {holdingMinutes}min</span>
                    </div>
                    <span className="text-muted-foreground">
                      Exit in: {timeToExit}min
                    </span>
                  </div>
                  
                  {/* Progress bar */}
                  <div className="w-full bg-muted rounded-full h-1.5">
                    <div 
                      className={`h-1.5 rounded-full transition-all duration-300 ${
                        holdingMinutes > 150 ? 'bg-trading-loss' :
                        holdingMinutes > 90 ? 'bg-trading-pending' : 'bg-trading-profit'
                      }`}
                      style={{ width: `${Math.min((holdingMinutes / 180) * 100, 100)}%` }}
                    />
                  </div>
                </div>

                {/* Risk Warnings */}
                {riskLevel === 'high' && (
                  <div className="flex items-center space-x-2 text-xs text-trading-loss bg-trading-loss/10 px-2 py-1 rounded">
                    <AlertTriangle className="h-3 w-3" />
                    <span>High risk: Consider manual exit</span>
                  </div>
                )}

                {/* 15-minute boundary indicator */}
                {holdingMinutes > 0 && (
                  <div className="text-xs text-muted-foreground">
                    Next boundary: {Math.ceil(holdingMinutes / 15) * 15}min
                    {((holdingMinutes % 15) >= 13) && (
                      <Badge variant="pending" className="ml-2 text-xs">
                        Boundary approaching
                      </Badge>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
        
        {activePositions.length > 0 && (
          <div className="pt-3 border-t border-border">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Total positions: {activePositions.length}</span>
              <span>
                Total unrealized: {formatCurrency(
                  activePositions.reduce((sum, pos) => sum + pos.unrealizedPnl, 0)
                )}
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
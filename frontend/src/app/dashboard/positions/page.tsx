'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { 
  Wallet, 
  TrendingUp, 
  TrendingDown,
  Search,
  Filter,
  Download,
  X,
  Clock,
  Target,
  AlertTriangle
} from 'lucide-react';
import { usePositions, useClosePosition } from '@/hooks/use-api';
import { formatCurrency, formatPercentage, getPnLColorClass, formatDateTime } from '@/lib/utils';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';
import { toast } from 'react-hot-toast';

export default function PositionsPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<'ALL' | 'ACTIVE' | 'CLOSED'>('ALL');
  
  const { data: positions, isLoading, error } = usePositions();
  const closePositionMutation = useClosePosition();

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold tracking-tight">Positions</h1>
        </div>
        <LoadingSkeleton className="h-96" />
      </div>
    );
  }

  // Mock data for demonstration - include both active and closed positions
  const allPositions = positions || [
    {
      id: '1',
      symbol: 'BTCUSDT',
      side: 'LONG' as const,
      quantity: 0.1,
      entryPrice: 43250.00,
      currentPrice: 43650.00,
      unrealizedPnl: 40.00,
      realizedPnl: 0,
      percentage: 0.92,
      margin: 1000,
      leverage: 1,
      markPrice: 43650.00,
      timestamp: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
      orders: [],
      status: 'ACTIVE' as const,
      maxDrawdown: -0.5,
      holdingTimeMinutes: 45,
      createdAt: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
      updatedAt: new Date().toISOString(),
      stopLoss: 42000.00,
      takeProfit: 44500.00,
    },
    {
      id: '2',
      symbol: 'ETHUSDT',
      side: 'LONG' as const,
      quantity: 1.5,
      entryPrice: 2650.00,
      currentPrice: 2645.80,
      unrealizedPnl: -6.30,
      realizedPnl: 0,
      percentage: -0.16,
      margin: 1000,
      leverage: 1,
      markPrice: 2645.80,
      timestamp: new Date(Date.now() - 28 * 60 * 1000).toISOString(),
      orders: [],
      status: 'ACTIVE' as const,
      maxDrawdown: -1.2,
      holdingTimeMinutes: 28,
      createdAt: new Date(Date.now() - 28 * 60 * 1000).toISOString(),
      updatedAt: new Date().toISOString(),
      stopLoss: 2580.00,
      takeProfit: 2720.00,
    },
    {
      id: '3',
      symbol: 'ADAUSDT',
      side: 'LONG' as const,
      quantity: 2000,
      entryPrice: 0.4800,
      currentPrice: 0.4825,
      unrealizedPnl: 50.00,
      realizedPnl: 45.20,
      percentage: 0.52,
      margin: 1000,
      leverage: 1,
      markPrice: 0.4825,
      timestamp: new Date(Date.now() - 72 * 60 * 1000).toISOString(),
      orders: [],
      status: 'CLOSED' as const,
      maxDrawdown: -0.8,
      holdingTimeMinutes: 72,
      createdAt: new Date(Date.now() - 72 * 60 * 1000).toISOString(),
      updatedAt: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
      stopLoss: 0.4680,
      takeProfit: 0.4920,
    },
  ];

  const filteredPositions = allPositions.filter(position => {
    const matchesSearch = position.symbol.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterStatus === 'ALL' || position.status === filterStatus;
    return matchesSearch && matchesFilter;
  });

  const activePositions = allPositions.filter(p => p.status === 'ACTIVE');
  const totalUnrealizedPnL = activePositions.reduce((sum, pos) => sum + pos.unrealizedPnl, 0);
  const totalPositionValue = activePositions.reduce((sum, pos) => sum + (pos.quantity * pos.currentPrice), 0);

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
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Positions</h1>
          <p className="text-muted-foreground">
            Manage and monitor all trading positions
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Active Positions</p>
                <p className="text-2xl font-bold">{activePositions.length}</p>
              </div>
              <Wallet className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Value</p>
                <p className="text-2xl font-bold">{formatCurrency(totalPositionValue)}</p>
              </div>
              <TrendingUp className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Unrealized P&L</p>
                <p className={`text-2xl font-bold ${getPnLColorClass(totalUnrealizedPnL)}`}>
                  {formatCurrency(totalUnrealizedPnL)}
                </p>
              </div>
              {totalUnrealizedPnL >= 0 ? (
                <TrendingUp className="h-8 w-8 text-trading-profit" />
              ) : (
                <TrendingDown className="h-8 w-8 text-trading-loss" />
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Avg Hold Time</p>
                <p className="text-2xl font-bold">
                  {activePositions.length > 0 
                    ? Math.round(activePositions.reduce((sum, pos) => sum + pos.holdingTimeMinutes, 0) / activePositions.length)
                    : 0}min
                </p>
              </div>
              <Clock className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters and Search */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Position History</CardTitle>
              <CardDescription>
                View and manage all trading positions
              </CardDescription>
            </div>
            
            <Badge variant="neutral" className="text-xs">
              {filteredPositions.length} positions
            </Badge>
          </div>
          
          <div className="flex items-center space-x-4 pt-4">
            <div className="relative flex-1 max-w-sm">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search symbol..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9"
              />
            </div>
            
            <div className="flex items-center space-x-2">
              {(['ALL', 'ACTIVE', 'CLOSED'] as const).map((status) => (
                <Button
                  key={status}
                  variant={filterStatus === status ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setFilterStatus(status)}
                  className="text-xs"
                >
                  {status}
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {filteredPositions.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Wallet className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="text-sm">No positions found</p>
              <p className="text-xs mt-1">Try adjusting your search criteria</p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Desktop Table */}
              <div className="hidden lg:block">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-muted-foreground">
                        <th className="text-left py-3">Symbol</th>
                        <th className="text-left py-3">Side</th>
                        <th className="text-right py-3">Quantity</th>
                        <th className="text-right py-3">Entry Price</th>
                        <th className="text-right py-3">Current Price</th>
                        <th className="text-right py-3">P&L</th>
                        <th className="text-center py-3">Hold Time</th>
                        <th className="text-center py-3">Risk</th>
                        <th className="text-center py-3">Status</th>
                        <th className="text-center py-3">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {filteredPositions.map((position) => {
                        const riskLevel = getPositionRiskLevel(position);
                        const timeToExit = Math.max(0, 180 - position.holdingTimeMinutes);
                        
                        return (
                          <tr key={position.id} className="hover:bg-muted/50 transition-colors">
                            <td className="py-4">
                              <div className="font-medium">{position.symbol}</div>
                              <div className="text-xs text-muted-foreground">
                                Margin: {formatCurrency(position.margin)}
                              </div>
                            </td>
                            
                            <td className="py-4">
                              <Badge 
                                variant={position.side === 'LONG' ? 'profit' : 'loss'}
                                className="text-xs"
                              >
                                {position.side}
                              </Badge>
                            </td>
                            
                            <td className="py-4 text-right font-mono">
                              {position.quantity}
                            </td>
                            
                            <td className="py-4 text-right font-mono">
                              {formatCurrency(position.entryPrice)}
                            </td>
                            
                            <td className="py-4 text-right font-mono">
                              {formatCurrency(position.currentPrice)}
                              <div className="text-xs text-muted-foreground">
                                {formatPercentage(position.percentage)}
                              </div>
                            </td>
                            
                            <td className="py-4 text-right">
                              <div className={`font-bold ${getPnLColorClass(position.unrealizedPnl)}`}>
                                {formatCurrency(position.unrealizedPnl)}
                              </div>
                              {position.realizedPnl !== 0 && (
                                <div className="text-xs text-muted-foreground">
                                  Realized: {formatCurrency(position.realizedPnl)}
                                </div>
                              )}
                            </td>
                            
                            <td className="py-4 text-center">
                              <div className="text-sm font-medium">
                                {position.holdingTimeMinutes}min
                              </div>
                              {position.status === 'ACTIVE' && (
                                <div className="text-xs text-muted-foreground">
                                  Exit: {timeToExit}min
                                </div>
                              )}
                            </td>
                            
                            <td className="py-4 text-center">
                              <Badge 
                                variant={getRiskBadgeVariant(riskLevel)}
                                className="text-xs capitalize"
                              >
                                {riskLevel}
                              </Badge>
                            </td>
                            
                            <td className="py-4 text-center">
                              <Badge 
                                variant={position.status === 'ACTIVE' ? 'profit' : 'neutral'}
                                className="text-xs"
                              >
                                {position.status}
                              </Badge>
                            </td>
                            
                            <td className="py-4 text-center">
                              {position.status === 'ACTIVE' && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleClosePosition(position.id, position.symbol)}
                                  disabled={closePositionMutation.isPending}
                                  className="h-8 w-8 p-0"
                                >
                                  <X className="h-3 w-3" />
                                </Button>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Mobile Cards */}
              <div className="lg:hidden space-y-4">
                {filteredPositions.map((position) => {
                  const riskLevel = getPositionRiskLevel(position);
                  const timeToExit = Math.max(0, 180 - position.holdingTimeMinutes);
                  
                  return (
                    <Card key={position.id} className="p-4">
                      <div className="space-y-4">
                        {/* Header */}
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <span className="font-semibold text-lg">{position.symbol}</span>
                            <Badge 
                              variant={position.side === 'LONG' ? 'profit' : 'loss'}
                              className="text-xs"
                            >
                              {position.side}
                            </Badge>
                            <Badge 
                              variant={position.status === 'ACTIVE' ? 'profit' : 'neutral'}
                              className="text-xs"
                            >
                              {position.status}
                            </Badge>
                          </div>
                          
                          {position.status === 'ACTIVE' && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleClosePosition(position.id, position.symbol)}
                              disabled={closePositionMutation.isPending}
                              className="h-8 w-8 p-0"
                            >
                              <X className="h-3 w-3" />
                            </Button>
                          )}
                        </div>

                        {/* Details */}
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <p className="text-muted-foreground text-xs">Quantity</p>
                            <p className="font-medium font-mono">{position.quantity}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground text-xs">Entry Price</p>
                            <p className="font-medium font-mono">{formatCurrency(position.entryPrice)}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground text-xs">Current Price</p>
                            <p className="font-medium font-mono">{formatCurrency(position.currentPrice)}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground text-xs">P&L</p>
                            <p className={`font-bold ${getPnLColorClass(position.unrealizedPnl)}`}>
                              {formatCurrency(position.unrealizedPnl)}
                            </p>
                          </div>
                          <div>
                            <p className="text-muted-foreground text-xs">Hold Time</p>
                            <p className="font-medium">{position.holdingTimeMinutes}min</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground text-xs">Risk Level</p>
                            <Badge 
                              variant={getRiskBadgeVariant(riskLevel)}
                              className="text-xs capitalize"
                            >
                              {riskLevel}
                            </Badge>
                          </div>
                        </div>

                        {/* Risk Warnings */}
                        {riskLevel === 'high' && position.status === 'ACTIVE' && (
                          <div className="flex items-center space-x-2 text-xs text-trading-loss bg-trading-loss/10 px-3 py-2 rounded">
                            <AlertTriangle className="h-3 w-3" />
                            <span>High risk position - consider manual exit</span>
                          </div>
                        )}

                        {/* Footer */}
                        <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t border-border">
                          <span>Created: {formatDateTime(position.createdAt)}</span>
                          {position.status === 'ACTIVE' && (
                            <span>Exit in: {timeToExit}min</span>
                          )}
                        </div>
                      </div>
                    </Card>
                  );
                })}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
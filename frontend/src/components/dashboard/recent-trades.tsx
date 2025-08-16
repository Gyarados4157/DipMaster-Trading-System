'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { 
  Activity,
  Search,
  Filter,
  ExternalLink,
  TrendingUp,
  TrendingDown,
  Clock,
  DollarSign
} from 'lucide-react';
import { useTrades } from '@/hooks/use-api';
import { formatCurrency, formatDateTime, getPnLColorClass, getTradeStatusColorClass } from '@/lib/utils';
import { LoadingSkeleton, TableSkeleton } from '@/components/ui/loading-skeleton';

export function RecentTrades() {
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const { data: tradesData, isLoading, error } = useTrades(currentPage, 20);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Recent Trades</CardTitle>
        </CardHeader>
        <CardContent>
          <TableSkeleton rows={10} />
        </CardContent>
      </Card>
    );
  }

  const trades = tradesData?.data || [];
  const pagination = tradesData?.pagination;

  const filteredTrades = trades.filter(trade =>
    trade.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    trade.status.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getStatusBadgeVariant = (status: string) => {
    switch (status.toLowerCase()) {
      case 'filled':
      case 'completed':
        return 'profit';
      case 'pending':
      case 'partially_filled':
        return 'pending';
      case 'cancelled':
      case 'rejected':
        return 'loss';
      default:
        return 'neutral';
    }
  };

  const getSideBadgeVariant = (side: string) => {
    return side.toLowerCase() === 'buy' ? 'profit' : 'loss';
  };

  const calculatePnL = (trade: any) => {
    if (trade.side === 'SELL' && trade.avgPrice && trade.executedQuantity) {
      // This is a simplified PnL calculation
      // In reality, you'd need to match with corresponding buy orders
      return trade.avgPrice * trade.executedQuantity * 0.01; // Mock 1% profit
    }
    return null;
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Recent Trades</span>
            </CardTitle>
            <CardDescription>
              Latest trading activity and execution history
            </CardDescription>
          </div>
          
          <Badge variant="neutral" className="text-xs">
            {trades.length} trades
          </Badge>
        </div>

        {/* Search and Filter */}
        <div className="flex items-center space-x-2 pt-4">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search symbol, status..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-9"
            />
          </div>
          <Button variant="outline" size="sm">
            <Filter className="h-4 w-4 mr-2" />
            Filter
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        {filteredTrades.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-sm">No trades found</p>
            <p className="text-xs mt-1">Try adjusting your search criteria</p>
          </div>
        ) : (
          <div className="space-y-3">
            {/* Desktop Table */}
            <div className="hidden md:block">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-muted-foreground">
                      <th className="text-left py-2">Symbol</th>
                      <th className="text-left py-2">Side</th>
                      <th className="text-right py-2">Quantity</th>
                      <th className="text-right py-2">Price</th>
                      <th className="text-right py-2">Value</th>
                      <th className="text-center py-2">Status</th>
                      <th className="text-right py-2">P&L</th>
                      <th className="text-right py-2">Time</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {filteredTrades.map((trade) => {
                      const pnl = calculatePnL(trade);
                      const totalValue = (trade.avgPrice || trade.price || 0) * trade.executedQuantity;
                      
                      return (
                        <tr key={trade.id} className="hover:bg-muted/50 transition-colors">
                          <td className="py-3">
                            <div className="font-medium">{trade.symbol}</div>
                            <div className="text-xs text-muted-foreground">
                              {trade.type}
                            </div>
                          </td>
                          <td className="py-3">
                            <Badge variant={getSideBadgeVariant(trade.side)} className="text-xs">
                              {trade.side}
                            </Badge>
                          </td>
                          <td className="py-3 text-right font-mono">
                            <div>{trade.executedQuantity}</div>
                            {trade.quantity !== trade.executedQuantity && (
                              <div className="text-xs text-muted-foreground">
                                of {trade.quantity}
                              </div>
                            )}
                          </td>
                          <td className="py-3 text-right font-mono">
                            {formatCurrency(trade.avgPrice || trade.price || 0)}
                          </td>
                          <td className="py-3 text-right font-mono">
                            {formatCurrency(totalValue)}
                          </td>
                          <td className="py-3 text-center">
                            <Badge variant={getStatusBadgeVariant(trade.status)} className="text-xs">
                              {trade.status}
                            </Badge>
                          </td>
                          <td className="py-3 text-right">
                            {pnl ? (
                              <span className={`font-medium ${getPnLColorClass(pnl)}`}>
                                {formatCurrency(pnl)}
                              </span>
                            ) : (
                              <span className="text-muted-foreground">-</span>
                            )}
                          </td>
                          <td className="py-3 text-right text-xs text-muted-foreground">
                            {formatDateTime(trade.createdAt)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Mobile Cards */}
            <div className="md:hidden space-y-3">
              {filteredTrades.map((trade) => {
                const pnl = calculatePnL(trade);
                const totalValue = (trade.avgPrice || trade.price || 0) * trade.executedQuantity;
                
                return (
                  <div
                    key={trade.id}
                    className="border border-border rounded-lg p-4 space-y-3 bg-card/50"
                  >
                    {/* Header */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="font-semibold">{trade.symbol}</span>
                        <Badge variant={getSideBadgeVariant(trade.side)} className="text-xs">
                          {trade.side}
                        </Badge>
                      </div>
                      <Badge variant={getStatusBadgeVariant(trade.status)} className="text-xs">
                        {trade.status}
                      </Badge>
                    </div>

                    {/* Details */}
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <p className="text-muted-foreground text-xs">Quantity</p>
                        <p className="font-medium font-mono">{trade.executedQuantity}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground text-xs">Price</p>
                        <p className="font-medium font-mono">
                          {formatCurrency(trade.avgPrice || trade.price || 0)}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground text-xs">Value</p>
                        <p className="font-medium font-mono">{formatCurrency(totalValue)}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground text-xs">P&L</p>
                        {pnl ? (
                          <p className={`font-medium ${getPnLColorClass(pnl)}`}>
                            {formatCurrency(pnl)}
                          </p>
                        ) : (
                          <p className="text-muted-foreground">-</p>
                        )}
                      </div>
                    </div>

                    {/* Footer */}
                    <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t border-border">
                      <span>{trade.type}</span>
                      <span>{formatDateTime(trade.createdAt)}</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Pagination */}
            {pagination && pagination.totalPages > 1 && (
              <div className="flex items-center justify-between pt-4 border-t border-border">
                <div className="text-sm text-muted-foreground">
                  Page {pagination.page} of {pagination.totalPages} 
                  ({pagination.total} total trades)
                </div>
                <div className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    disabled={!pagination.hasPrev || isLoading}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(currentPage + 1)}
                    disabled={!pagination.hasNext || isLoading}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  BarChart3,
  RefreshCw,
  Eye,
  EyeOff
} from 'lucide-react';
import { useMarketData } from '@/hooks/use-api';
import { formatCurrency, formatPercentage, getPnLColorClass, formatNumber } from '@/lib/utils';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';
import { useState } from 'react';

export function MarketOverview() {
  const [showAllSymbols, setShowAllSymbols] = useState(false);
  const { data: marketData, isLoading, error, refetch } = useMarketData();

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Market Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingSkeleton lines={5} className="h-80" />
        </CardContent>
      </Card>
    );
  }

  // Mock data if no real data available
  const symbols = marketData || [
    {
      symbol: 'BTCUSDT',
      baseAsset: 'BTC',
      quoteAsset: 'USDT',
      status: 'TRADING' as const,
      price: 43650.00,
      priceChange: 425.50,
      priceChangePercent: 0.98,
      volume: 15432.45,
      quoteVolume: 673542180.25,
      high24h: 44250.00,
      low24h: 42980.00,
      lastUpdate: new Date().toISOString(),
    },
    {
      symbol: 'ETHUSDT',
      baseAsset: 'ETH',
      quoteAsset: 'USDT',
      status: 'TRADING' as const,
      price: 2645.80,
      priceChange: -15.20,
      priceChangePercent: -0.57,
      volume: 45621.22,
      quoteVolume: 120654320.45,
      high24h: 2698.50,
      low24h: 2590.10,
      lastUpdate: new Date().toISOString(),
    },
    {
      symbol: 'ADAUSDT',
      baseAsset: 'ADA',
      quoteAsset: 'USDT',
      status: 'TRADING' as const,
      price: 0.4825,
      priceChange: 0.0123,
      priceChangePercent: 2.62,
      volume: 156432890.45,
      quoteVolume: 75432180.25,
      high24h: 0.4950,
      low24h: 0.4680,
      lastUpdate: new Date().toISOString(),
    },
    {
      symbol: 'SOLUSDT',
      baseAsset: 'SOL',
      quoteAsset: 'USDT',
      status: 'TRADING' as const,
      price: 98.45,
      priceChange: 3.25,
      priceChangePercent: 3.42,
      volume: 25432.15,
      quoteVolume: 2500542.80,
      high24h: 102.50,
      low24h: 94.20,
      lastUpdate: new Date().toISOString(),
    },
    {
      symbol: 'DOGEUSDT',
      baseAsset: 'DOGE',
      quoteAsset: 'USDT',
      status: 'TRADING' as const,
      price: 0.08234,
      priceChange: -0.00156,
      priceChangePercent: -1.86,
      volume: 2543218945.25,
      quoteVolume: 209432180.45,
      high24h: 0.08450,
      low24h: 0.08120,
      lastUpdate: new Date().toISOString(),
    },
  ];

  const displaySymbols = showAllSymbols ? symbols : symbols.slice(0, 3);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'TRADING': return 'profit';
      case 'HALT': return 'pending';
      case 'BREAK': return 'loss';
      default: return 'neutral';
    }
  };

  const getTrendIcon = (change: number) => {
    return change >= 0 ? TrendingUp : TrendingDown;
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Market Overview</span>
            </CardTitle>
            <CardDescription>
              Real-time market data for trading pairs
            </CardDescription>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => refetch()}
              className="h-8 w-8 p-0"
            >
              <RefreshCw className="h-3 w-3" />
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowAllSymbols(!showAllSymbols)}
              className="h-8 px-3"
            >
              {showAllSymbols ? (
                <>
                  <EyeOff className="h-3 w-3 mr-1" />
                  Less
                </>
              ) : (
                <>
                  <Eye className="h-3 w-3 mr-1" />
                  All
                </>
              )}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <div className="space-y-3">
          {/* Desktop Table */}
          <div className="hidden md:block">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-muted-foreground">
                    <th className="text-left py-2">Symbol</th>
                    <th className="text-right py-2">Price</th>
                    <th className="text-right py-2">24h Change</th>
                    <th className="text-right py-2">Volume</th>
                    <th className="text-center py-2">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {displaySymbols.map((symbol) => {
                    const TrendIcon = getTrendIcon(symbol.priceChange);
                    
                    return (
                      <tr key={symbol.symbol} className="hover:bg-muted/50 transition-colors">
                        <td className="py-3">
                          <div className="flex items-center space-x-2">
                            <div>
                              <div className="font-medium">{symbol.symbol}</div>
                              <div className="text-xs text-muted-foreground">
                                {symbol.baseAsset}/{symbol.quoteAsset}
                              </div>
                            </div>
                          </div>
                        </td>
                        
                        <td className="py-3 text-right">
                          <div className="font-mono font-medium">
                            {formatCurrency(symbol.price)}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            H: {formatCurrency(symbol.high24h)} L: {formatCurrency(symbol.low24h)}
                          </div>
                        </td>
                        
                        <td className="py-3 text-right">
                          <div className={`flex items-center justify-end space-x-1 ${getPnLColorClass(symbol.priceChange)}`}>
                            <TrendIcon className="h-3 w-3" />
                            <span className="font-medium">
                              {formatCurrency(symbol.priceChange)}
                            </span>
                          </div>
                          <div className={`text-xs ${getPnLColorClass(symbol.priceChange)}`}>
                            {symbol.priceChangePercent >= 0 ? '+' : ''}
                            {formatPercentage(symbol.priceChangePercent)}
                          </div>
                        </td>
                        
                        <td className="py-3 text-right">
                          <div className="font-mono text-sm">
                            {formatNumber(symbol.volume)}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {formatNumber(symbol.quoteVolume)} USDT
                          </div>
                        </td>
                        
                        <td className="py-3 text-center">
                          <Badge variant={getStatusColor(symbol.status)} className="text-xs">
                            {symbol.status}
                          </Badge>
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
            {displaySymbols.map((symbol) => {
              const TrendIcon = getTrendIcon(symbol.priceChange);
              
              return (
                <div
                  key={symbol.symbol}
                  className="border border-border rounded-lg p-4 space-y-3 bg-card/50"
                >
                  {/* Header */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold">{symbol.symbol}</span>
                      <Badge variant={getStatusColor(symbol.status)} className="text-xs">
                        {symbol.status}
                      </Badge>
                    </div>
                    
                    <div className="text-right">
                      <div className="font-mono font-medium">
                        {formatCurrency(symbol.price)}
                      </div>
                    </div>
                  </div>

                  {/* Details */}
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <p className="text-muted-foreground text-xs">24h Change</p>
                      <div className={`flex items-center space-x-1 ${getPnLColorClass(symbol.priceChange)}`}>
                        <TrendIcon className="h-3 w-3" />
                        <span className="font-medium">
                          {formatCurrency(symbol.priceChange)}
                        </span>
                        <span>
                          ({symbol.priceChangePercent >= 0 ? '+' : ''}
                          {formatPercentage(symbol.priceChangePercent)})
                        </span>
                      </div>
                    </div>
                    
                    <div>
                      <p className="text-muted-foreground text-xs">Volume (24h)</p>
                      <p className="font-medium font-mono">{formatNumber(symbol.volume)}</p>
                    </div>
                    
                    <div>
                      <p className="text-muted-foreground text-xs">High (24h)</p>
                      <p className="font-medium font-mono">{formatCurrency(symbol.high24h)}</p>
                    </div>
                    
                    <div>
                      <p className="text-muted-foreground text-xs">Low (24h)</p>
                      <p className="font-medium font-mono">{formatCurrency(symbol.low24h)}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Summary Stats */}
          <div className="pt-4 border-t border-border">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div className="p-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-muted-foreground">Avg Change</p>
                <p className={`text-sm font-bold ${getPnLColorClass(
                  symbols.reduce((sum, s) => sum + s.priceChangePercent, 0) / symbols.length
                )}`}>
                  {formatPercentage(
                    symbols.reduce((sum, s) => sum + s.priceChangePercent, 0) / symbols.length
                  )}
                </p>
              </div>
              
              <div className="p-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-muted-foreground">Total Volume</p>
                <p className="text-sm font-bold">
                  {formatNumber(
                    symbols.reduce((sum, s) => sum + s.quoteVolume, 0)
                  )}
                </p>
              </div>
              
              <div className="p-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-muted-foreground">Active Pairs</p>
                <p className="text-sm font-bold text-trading-profit">
                  {symbols.filter(s => s.status === 'TRADING').length}
                </p>
              </div>
              
              <div className="p-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-muted-foreground">Last Update</p>
                <p className="text-sm font-bold text-muted-foreground">
                  {new Date().toLocaleTimeString()}
                </p>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
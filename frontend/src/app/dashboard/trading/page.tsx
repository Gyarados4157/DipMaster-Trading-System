'use client';

import { Suspense } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  TrendingUp, 
  Activity, 
  Zap,
  Eye,
  Settings,
  RefreshCw
} from 'lucide-react';
import { ActivePositions } from '@/components/dashboard/active-positions';
import { MarketOverview } from '@/components/dashboard/market-overview';
import { RecentTrades } from '@/components/dashboard/recent-trades';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';

export default function TradingPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Live Trading</h1>
          <p className="text-muted-foreground">
            Real-time trading operations and market monitoring
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Badge variant="profit" className="animate-pulse">
            <div className="w-2 h-2 bg-white rounded-full mr-2" />
            Strategy Active
          </Badge>
          
          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* Strategy Status */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center space-x-2">
                <Zap className="h-5 w-5 text-dipmaster-blue" />
                <span>DipMaster Strategy Status</span>
              </CardTitle>
              <CardDescription>
                Real-time strategy monitoring and controls
              </CardDescription>
            </div>
            
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
              <Button variant="outline" size="sm">
                <Eye className="h-4 w-4 mr-2" />
                Monitor
              </Button>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-trading-profit/10 rounded-lg border border-trading-profit/20">
              <div className="text-2xl font-bold text-trading-profit">ACTIVE</div>
              <div className="text-sm text-muted-foreground">Strategy Status</div>
              <div className="mt-2">
                <Badge variant="profit" className="text-xs">
                  Monitoring Markets
                </Badge>
              </div>
            </div>
            
            <div className="text-center p-4 bg-dipmaster-blue/10 rounded-lg border border-dipmaster-blue/20">
              <div className="text-2xl font-bold text-dipmaster-blue">3/3</div>
              <div className="text-sm text-muted-foreground">Active Signals</div>
              <div className="mt-2">
                <Badge variant="neutral" className="text-xs">
                  BTCUSDT, ETHUSDT, ADAUSDT
                </Badge>
              </div>
            </div>
            
            <div className="text-center p-4 bg-dipmaster-orange/10 rounded-lg border border-dipmaster-orange/20">
              <div className="text-2xl font-bold text-dipmaster-orange">96min</div>
              <div className="text-sm text-muted-foreground">Avg Position Time</div>
              <div className="mt-2">
                <Badge variant="pending" className="text-xs">
                  Within Target
                </Badge>
              </div>
            </div>
            
            <div className="text-center p-4 bg-dipmaster-green/10 rounded-lg border border-dipmaster-green/20">
              <div className="text-2xl font-bold text-dipmaster-green">100%</div>
              <div className="text-sm text-muted-foreground">Boundary Compliance</div>
              <div className="mt-2">
                <Badge variant="profit" className="text-xs">
                  Perfect Execution
                </Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Trading Interface */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Column - Market and Positions */}
        <div className="xl:col-span-2 space-y-6">
          {/* Market Overview */}
          <Suspense fallback={<LoadingSkeleton className="h-96" />}>
            <MarketOverview />
          </Suspense>

          {/* Trading Signals */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Activity className="h-5 w-5" />
                <span>Signal Monitor</span>
              </CardTitle>
              <CardDescription>
                Real-time entry and exit signal detection
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Signal Status */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-3 bg-muted/50 rounded-lg text-center">
                    <div className="text-lg font-bold text-trading-profit">15</div>
                    <div className="text-xs text-muted-foreground">Signals Today</div>
                  </div>
                  <div className="p-3 bg-muted/50 rounded-lg text-center">
                    <div className="text-lg font-bold text-dipmaster-blue">87.9%</div>
                    <div className="text-xs text-muted-foreground">Dip Buy Rate</div>
                  </div>
                  <div className="p-3 bg-muted/50 rounded-lg text-center">
                    <div className="text-lg font-bold text-dipmaster-orange">45s</div>
                    <div className="text-xs text-muted-foreground">Avg Response</div>
                  </div>
                </div>

                {/* Recent Signals */}
                <div className="space-y-2">
                  <h4 className="text-sm font-medium">Recent Signals</h4>
                  <div className="space-y-2">
                    {[
                      { symbol: 'BTCUSDT', type: 'ENTRY', time: '2 minutes ago', price: 43250.00, confidence: 0.89 },
                      { symbol: 'ETHUSDT', type: 'EXIT', time: '5 minutes ago', price: 2645.80, confidence: 0.92 },
                      { symbol: 'ADAUSDT', type: 'ENTRY', time: '8 minutes ago', price: 0.4825, confidence: 0.85 },
                    ].map((signal, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-card/50 rounded border">
                        <div className="flex items-center space-x-3">
                          <Badge variant={signal.type === 'ENTRY' ? 'profit' : 'loss'} className="text-xs">
                            {signal.type}
                          </Badge>
                          <div>
                            <div className="font-medium text-sm">{signal.symbol}</div>
                            <div className="text-xs text-muted-foreground">{signal.time}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-mono">${signal.price.toLocaleString()}</div>
                          <div className="text-xs text-muted-foreground">
                            Confidence: {(signal.confidence * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Positions and Controls */}
        <div className="space-y-6">
          {/* Active Positions */}
          <Suspense fallback={<LoadingSkeleton className="h-80" />}>
            <ActivePositions />
          </Suspense>

          {/* Strategy Controls */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Settings className="h-5 w-5" />
                <span>Strategy Controls</span>
              </CardTitle>
              <CardDescription>
                Manual controls and overrides
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-2">
                <Button variant="outline" size="sm" className="w-full">
                  Pause Strategy
                </Button>
                <Button variant="outline" size="sm" className="w-full">
                  Force Exit All
                </Button>
                <Button variant="outline" size="sm" className="w-full">
                  Skip Next Signal
                </Button>
                <Button variant="outline" size="sm" className="w-full">
                  Adjust Risk
                </Button>
              </div>
              
              <div className="pt-4 border-t border-border">
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Max Positions:</span>
                    <span className="font-medium">3</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Position Size:</span>
                    <span className="font-medium">$1,000</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Risk Level:</span>
                    <Badge variant="profit" className="text-xs">LOW</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Auto Trading:</span>
                    <Badge variant="profit" className="text-xs">ENABLED</Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Performance Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Today's Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Trades Executed</span>
                  <span className="font-bold">15</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Win Rate</span>
                  <span className="font-bold text-trading-profit">86.7%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Total P&L</span>
                  <span className="font-bold text-trading-profit">+$324.50</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Avg Hold Time</span>
                  <span className="font-bold">94min</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Recent Trading Activity */}
      <Suspense fallback={<LoadingSkeleton className="h-96" />}>
        <RecentTrades />
      </Suspense>
    </div>
  );
}
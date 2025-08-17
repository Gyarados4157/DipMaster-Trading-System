'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  PieChart, 
  Pie, 
  Cell, 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip,
  Legend
} from 'recharts';
import { 
  Wallet, 
  TrendingUp, 
  TrendingDown, 
  Target,
  Globe,
  BarChart3
} from 'lucide-react';

interface ExposureData {
  category: string;
  value: number;
  percentage: number;
  color: string;
  risk: 'low' | 'medium' | 'high';
}

interface AssetExposure {
  symbol: string;
  allocation: number;
  value: number;
  risk: number;
  beta: number;
  sector: string;
}

export function PortfolioExposure() {
  const sectorExposure: ExposureData[] = [
    { category: 'Layer 1', value: 45000, percentage: 45, color: '#3b82f6', risk: 'medium' },
    { category: 'DeFi', value: 25000, percentage: 25, color: '#10b981', risk: 'high' },
    { category: 'Infrastructure', value: 15000, percentage: 15, color: '#f59e0b', risk: 'low' },
    { category: 'Gaming/NFT', value: 8000, percentage: 8, color: '#ef4444', risk: 'high' },
    { category: 'Stablecoin', value: 7000, percentage: 7, color: '#6b7280', risk: 'low' },
  ];

  const geographicExposure: ExposureData[] = [
    { category: '全球', value: 60000, percentage: 60, color: '#3b82f6', risk: 'medium' },
    { category: '美国', value: 20000, percentage: 20, color: '#10b981', risk: 'low' },
    { category: '欧洲', value: 12000, percentage: 12, color: '#f59e0b', risk: 'low' },
    { category: '亚洲', value: 8000, percentage: 8, color: '#ef4444', risk: 'medium' },
  ];

  const assetExposures: AssetExposure[] = [
    { symbol: 'BTC', allocation: 25, value: 25000, risk: 0.65, beta: 1.0, sector: 'Layer 1' },
    { symbol: 'ETH', allocation: 20, value: 20000, risk: 0.75, beta: 1.2, sector: 'Layer 1' },
    { symbol: 'SOL', allocation: 15, value: 15000, risk: 0.85, beta: 1.5, sector: 'Layer 1' },
    { symbol: 'UNI', allocation: 12, value: 12000, risk: 0.95, beta: 1.8, sector: 'DeFi' },
    { symbol: 'AAVE', allocation: 10, value: 10000, risk: 0.90, beta: 1.6, sector: 'DeFi' },
    { symbol: 'LINK', allocation: 8, value: 8000, risk: 0.70, beta: 1.1, sector: 'Infrastructure' },
    { symbol: 'ADA', allocation: 6, value: 6000, risk: 0.80, beta: 1.3, sector: 'Layer 1' },
    { symbol: 'DOT', allocation: 4, value: 4000, risk: 0.75, beta: 1.2, sector: 'Infrastructure' },
  ];

  const getRiskColor = (risk: 'low' | 'medium' | 'high') => {
    switch (risk) {
      case 'low':
        return 'text-trading-profit';
      case 'medium':
        return 'text-trading-pending';
      case 'high':
        return 'text-trading-loss';
      default:
        return 'text-muted-foreground';
    }
  };

  const getRiskBadge = (risk: 'low' | 'medium' | 'high') => {
    switch (risk) {
      case 'low':
        return <Badge variant="profit" className="text-xs">低风险</Badge>;
      case 'medium':
        return <Badge variant="pending" className="text-xs">中风险</Badge>;
      case 'high':
        return <Badge variant="loss" className="text-xs">高风险</Badge>;
      default:
        return <Badge variant="outline" className="text-xs">未知</Badge>;
    }
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-background border rounded-lg p-3 shadow-lg">
          <p className="font-medium">{data.category}</p>
          <p className="text-sm text-muted-foreground">
            价值: ${data.value.toLocaleString()}
          </p>
          <p className="text-sm text-muted-foreground">
            占比: {data.percentage}%
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* 板块分布 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg font-semibold">板块风险暴露</CardTitle>
              <CardDescription>
                不同加密货币板块的投资分布
              </CardDescription>
            </div>
            <BarChart3 className="h-5 w-5 text-muted-foreground" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={sectorExposure}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                  >
                    {sectorExposure.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            
            <div className="space-y-3">
              {sectorExposure.map((sector, index) => (
                <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: sector.color }}
                    />
                    <div>
                      <div className="font-medium text-sm">{sector.category}</div>
                      <div className="text-xs text-muted-foreground">
                        ${sector.value.toLocaleString()}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm font-medium">{sector.percentage}%</span>
                    {getRiskBadge(sector.risk)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 地域分布 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg font-semibold">地域风险暴露</CardTitle>
              <CardDescription>
                按地理位置分布的投资风险
              </CardDescription>
            </div>
            <Globe className="h-5 w-5 text-muted-foreground" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {geographicExposure.map((region, index) => (
              <div key={index} className="space-y-2">
                <div className="flex justify-between items-center">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-sm">{region.category}</span>
                    {getRiskBadge(region.risk)}
                  </div>
                  <div className="text-sm">
                    <span className="font-medium">${region.value.toLocaleString()}</span>
                    <span className="text-muted-foreground ml-2">({region.percentage}%)</span>
                  </div>
                </div>
                <Progress value={region.percentage} className="h-2" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 资产明细 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg font-semibold">资产风险明细</CardTitle>
              <CardDescription>
                单个资产的风险评估和Beta系数
              </CardDescription>
            </div>
            <Target className="h-5 w-5 text-muted-foreground" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {assetExposures.map((asset, index) => (
              <div key={index} className="p-3 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    <div className="font-medium text-sm">{asset.symbol}</div>
                    <Badge variant="outline" className="text-xs">{asset.sector}</Badge>
                  </div>
                  <div className="text-sm font-medium">
                    ${asset.value.toLocaleString()} ({asset.allocation}%)
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div>
                    <span className="text-muted-foreground">风险评分: </span>
                    <span className={
                      asset.risk > 0.8 ? 'text-trading-loss' : 
                      asset.risk > 0.6 ? 'text-trading-pending' : 
                      'text-trading-profit'
                    }>
                      {(asset.risk * 100).toFixed(0)}/100
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Beta系数: </span>
                    <span className={asset.beta > 1.2 ? 'text-trading-loss' : 'text-foreground'}>
                      {asset.beta.toFixed(1)}
                    </span>
                  </div>
                </div>
                
                <Progress value={asset.allocation} className="h-1 mt-2" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
'use client';

import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { 
  Filter, 
  Search, 
  X, 
  Calendar,
  AlertCircle,
  AlertTriangle,
  Info,
  CheckCircle2
} from 'lucide-react';

interface FilterState {
  search: string;
  types: string[];
  categories: string[];
  sources: string[];
  timeRange: string;
  acknowledged: 'all' | 'acknowledged' | 'unacknowledged';
}

export function AlertFilters() {
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    types: [],
    categories: [],
    sources: [],
    timeRange: '24h',
    acknowledged: 'all',
  });

  const [showAdvanced, setShowAdvanced] = useState(false);

  const alertTypes = [
    { value: 'critical', label: 'Critical', icon: AlertCircle, color: 'text-trading-loss' },
    { value: 'warning', label: 'Warning', icon: AlertTriangle, color: 'text-trading-pending' },
    { value: 'info', label: 'Info', icon: Info, color: 'text-dipmaster-blue' },
    { value: 'resolved', label: 'Resolved', icon: CheckCircle2, color: 'text-trading-profit' },
  ];

  const categories = [
    'Risk Management',
    'Market Analysis',
    'System',
    'Performance',
    'Trading',
    'Connectivity',
  ];

  const sources = [
    'Risk Monitor',
    'Signal Detector',
    'WebSocket Monitor',
    'Strategy Monitor',
    'Position Manager',
    'Order Executor',
  ];

  const timeRanges = [
    { value: '1h', label: 'Last Hour' },
    { value: '24h', label: 'Last 24 Hours' },
    { value: '7d', label: 'Last Week' },
    { value: '30d', label: 'Last Month' },
    { value: 'custom', label: 'Custom Range' },
  ];

  const handleTypeToggle = (type: string) => {
    setFilters(prev => ({
      ...prev,
      types: prev.types.includes(type)
        ? prev.types.filter(t => t !== type)
        : [...prev.types, type]
    }));
  };

  const handleCategoryToggle = (category: string) => {
    setFilters(prev => ({
      ...prev,
      categories: prev.categories.includes(category)
        ? prev.categories.filter(c => c !== category)
        : [...prev.categories, category]
    }));
  };

  const handleSourceToggle = (source: string) => {
    setFilters(prev => ({
      ...prev,
      sources: prev.sources.includes(source)
        ? prev.sources.filter(s => s !== source)
        : [...prev.sources, source]
    }));
  };

  const clearAllFilters = () => {
    setFilters({
      search: '',
      types: [],
      categories: [],
      sources: [],
      timeRange: '24h',
      acknowledged: 'all',
    });
  };

  const getActiveFilterCount = () => {
    return filters.types.length + 
           filters.categories.length + 
           filters.sources.length + 
           (filters.search ? 1 : 0) +
           (filters.acknowledged !== 'all' ? 1 : 0) +
           (filters.timeRange !== '24h' ? 1 : 0);
  };

  const activeFilterCount = getActiveFilterCount();

  return (
    <Card className="p-4">
      <div className="space-y-4">
        {/* Search and Basic Filters */}
        <div className="flex items-center space-x-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search alerts..."
              value={filters.search}
              onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
              className="pl-10"
            />
          </div>
          
          <div className="flex items-center space-x-2">
            {timeRanges.map((range) => (
              <Button
                key={range.value}
                variant={filters.timeRange === range.value ? 'default' : 'outline'}
                size="sm"
                onClick={() => setFilters(prev => ({ ...prev, timeRange: range.value }))}
                className="text-xs"
              >
                {range.label}
              </Button>
            ))}
          </div>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs"
          >
            <Filter className="h-3 w-3 mr-1" />
            Advanced
            {activeFilterCount > 0 && (
              <Badge variant="secondary" className="ml-2 text-xs">
                {activeFilterCount}
              </Badge>
            )}
          </Button>
        </div>

        {/* Alert Type Quick Filters */}
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium text-muted-foreground">Type:</span>
          {alertTypes.map((type) => {
            const Icon = type.icon;
            const isSelected = filters.types.includes(type.value);
            
            return (
              <Button
                key={type.value}
                variant={isSelected ? 'default' : 'outline'}
                size="sm"
                onClick={() => handleTypeToggle(type.value)}
                className="text-xs"
              >
                <Icon className={`h-3 w-3 mr-1 ${isSelected ? 'text-background' : type.color}`} />
                {type.label}
              </Button>
            );
          })}
        </div>

        {/* Advanced Filters */}
        {showAdvanced && (
          <div className="space-y-4 pt-4 border-t">
            {/* Acknowledgment Status */}
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium text-muted-foreground min-w-20">Status:</span>
              <div className="flex items-center space-x-1">
                {[
                  { value: 'all', label: 'All' },
                  { value: 'acknowledged', label: 'Acknowledged' },
                  { value: 'unacknowledged', label: 'Unacknowledged' },
                ].map((option) => (
                  <Button
                    key={option.value}
                    variant={filters.acknowledged === option.value ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setFilters(prev => ({ ...prev, acknowledged: option.value as any }))}
                    className="text-xs"
                  >
                    {option.label}
                  </Button>
                ))}
              </div>
            </div>

            {/* Categories */}
            <div className="flex items-start space-x-2">
              <span className="text-sm font-medium text-muted-foreground min-w-20 pt-1">Category:</span>
              <div className="flex flex-wrap gap-1">
                {categories.map((category) => {
                  const isSelected = filters.categories.includes(category);
                  
                  return (
                    <Button
                      key={category}
                      variant={isSelected ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => handleCategoryToggle(category)}
                      className="text-xs h-7"
                    >
                      {category}
                      {isSelected && <X className="h-3 w-3 ml-1" />}
                    </Button>
                  );
                })}
              </div>
            </div>

            {/* Sources */}
            <div className="flex items-start space-x-2">
              <span className="text-sm font-medium text-muted-foreground min-w-20 pt-1">Source:</span>
              <div className="flex flex-wrap gap-1">
                {sources.map((source) => {
                  const isSelected = filters.sources.includes(source);
                  
                  return (
                    <Button
                      key={source}
                      variant={isSelected ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => handleSourceToggle(source)}
                      className="text-xs h-7"
                    >
                      {source}
                      {isSelected && <X className="h-3 w-3 ml-1" />}
                    </Button>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Active Filters Summary */}
        {activeFilterCount > 0 && (
          <div className="flex items-center justify-between pt-4 border-t">
            <div className="flex items-center space-x-2">
              <span className="text-sm text-muted-foreground">Active filters:</span>
              <Badge variant="secondary" className="text-xs">
                {activeFilterCount} filter{activeFilterCount > 1 ? 's' : ''} applied
              </Badge>
            </div>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={clearAllFilters}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              Clear all filters
            </Button>
          </div>
        )}
      </div>
    </Card>
  );
}
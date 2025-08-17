'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  exportToCSV, 
  exportToJSON, 
  exportToExcel, 
  exportConfigs, 
  generateFilename,
  validateExport,
  exportMultipleDatasets,
  formatters
} from '@/utils/export';
import { 
  Download, 
  FileText, 
  FileSpreadsheet, 
  Database,
  AlertCircle,
  CheckCircle2,
  Clock,
  X,
  Settings
} from 'lucide-react';

interface ExportManagerProps {
  data: any[];
  dataType: 'trades' | 'positions' | 'pnl' | 'alerts' | 'riskMetrics' | 'custom';
  customConfig?: Array<{ key: string; label: string; formatter?: (value: any) => string }>;
  title?: string;
  className?: string;
  compact?: boolean;
  onExportStart?: () => void;
  onExportComplete?: (format: string, filename: string) => void;
  onExportError?: (error: string) => void;
}

type ExportFormat = 'csv' | 'json' | 'excel';

interface ExportJob {
  id: string;
  format: ExportFormat;
  filename: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  error?: string;
  startTime: Date;
  progress?: number;
}

export function ExportManager({
  data,
  dataType,
  customConfig,
  title,
  className,
  compact = false,
  onExportStart,
  onExportComplete,
  onExportError
}: ExportManagerProps) {
  const [isExporting, setIsExporting] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [exportJobs, setExportJobs] = useState<ExportJob[]>([]);
  const [selectedFormats, setSelectedFormats] = useState<ExportFormat[]>(['csv']);

  const config = dataType === 'custom' ? customConfig : exportConfigs[dataType];
  const validation = validateExport(data);

  const exportFormats: Array<{
    format: ExportFormat;
    label: string;
    icon: React.ComponentType<any>;
    description: string;
  }> = [
    {
      format: 'csv',
      label: 'CSV',
      icon: FileText,
      description: 'Comma-separated values for spreadsheets'
    },
    {
      format: 'excel',
      label: 'Excel',
      icon: FileSpreadsheet,
      description: 'Microsoft Excel format'
    },
    {
      format: 'json',
      label: 'JSON',
      icon: Database,
      description: 'JavaScript Object Notation for developers'
    }
  ];

  const handleExport = async (format: ExportFormat) => {
    if (!validation.canExport) {
      onExportError?.(validation.warning || 'Cannot export data');
      return;
    }

    const jobId = `export_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const filename = generateFilename(dataType, format);
    
    const job: ExportJob = {
      id: jobId,
      format,
      filename,
      status: 'processing',
      startTime: new Date(),
      progress: 0
    };

    setExportJobs(prev => [job, ...prev]);
    setIsExporting(true);
    onExportStart?.();

    try {
      // Simulate progress for large datasets
      if (data.length > 1000) {
        // Update progress
        const progressInterval = setInterval(() => {
          setExportJobs(prev => prev.map(j => 
            j.id === jobId 
              ? { ...j, progress: Math.min((j.progress || 0) + 20, 90) }
              : j
          ));
        }, 200);

        // Wait a bit to show progress
        await new Promise(resolve => setTimeout(resolve, 1000));
        clearInterval(progressInterval);
      }

      // Perform the actual export
      switch (format) {
        case 'csv':
          exportToCSV(data, filename, config);
          break;
        case 'json':
          exportToJSON(data, filename, true);
          break;
        case 'excel':
          exportToExcel(data, filename, title || dataType, config);
          break;
      }

      // Mark job as completed
      setExportJobs(prev => prev.map(j => 
        j.id === jobId 
          ? { ...j, status: 'completed', progress: 100 }
          : j
      ));

      onExportComplete?.(format, filename);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Export failed';
      
      setExportJobs(prev => prev.map(j => 
        j.id === jobId 
          ? { ...j, status: 'error', error: errorMessage }
          : j
      ));

      onExportError?.(errorMessage);
    } finally {
      setIsExporting(false);
    }
  };

  const handleBatchExport = async () => {
    if (selectedFormats.length === 0) return;

    setIsExporting(true);
    onExportStart?.();

    try {
      for (const format of selectedFormats) {
        await handleExport(format);
        // Small delay between exports
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    } finally {
      setIsExporting(false);
    }
  };

  const removeJob = (jobId: string) => {
    setExportJobs(prev => prev.filter(j => j.id !== jobId));
  };

  const getStatusIcon = (status: ExportJob['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="h-4 w-4 text-trading-profit" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-trading-loss" />;
      case 'processing':
        return <Clock className="h-4 w-4 text-trading-pending animate-spin" />;
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  if (compact) {
    return (
      <div className={`relative ${className}`}>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowOptions(!showOptions)}
          disabled={!validation.canExport || isExporting}
          className="text-xs"
        >
          <Download className="h-3 w-3 mr-1" />
          Export
          {validation.warning && (
            <AlertCircle className="h-3 w-3 ml-1 text-trading-pending" />
          )}
        </Button>

        {showOptions && (
          <>
            <div 
              className="fixed inset-0 z-40"
              onClick={() => setShowOptions(false)}
            />
            <div className="absolute right-0 top-10 z-50 w-48 rounded-lg border border-border bg-popover p-2 shadow-lg">
              <div className="space-y-1">
                {exportFormats.map(({ format, label, icon: Icon }) => (
                  <button
                    key={format}
                    onClick={() => {
                      handleExport(format);
                      setShowOptions(false);
                    }}
                    disabled={isExporting}
                    className="flex w-full items-center space-x-2 rounded-md px-3 py-2 text-sm hover:bg-accent hover:text-accent-foreground transition-colors disabled:opacity-50"
                  >
                    <Icon className="h-4 w-4" />
                    <span>Export as {label}</span>
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    );
  }

  return (
    <Card className={`p-4 ${className}`}>
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold">Export Data</h3>
            <p className="text-sm text-muted-foreground">
              {data.length.toLocaleString()} rows • {title || dataType}
            </p>
          </div>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowOptions(!showOptions)}
          >
            <Settings className="h-4 w-4 mr-1" />
            Options
          </Button>
        </div>

        {/* Validation Warning */}
        {validation.warning && (
          <div className="flex items-center space-x-2 p-3 bg-trading-pending/10 border border-trading-pending/20 rounded-lg">
            <AlertCircle className="h-4 w-4 text-trading-pending" />
            <span className="text-sm">{validation.warning}</span>
          </div>
        )}

        {/* Export Options */}
        {showOptions && (
          <div className="space-y-3 p-3 bg-muted/30 rounded-lg border">
            <div>
              <label className="text-sm font-medium mb-2 block">Export Formats:</label>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                {exportFormats.map(({ format, label, icon: Icon, description }) => (
                  <label
                    key={format}
                    className="flex items-center space-x-2 cursor-pointer p-2 rounded border hover:bg-accent transition-colors"
                  >
                    <input
                      type="checkbox"
                      checked={selectedFormats.includes(format)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedFormats(prev => [...prev, format]);
                        } else {
                          setSelectedFormats(prev => prev.filter(f => f !== format));
                        }
                      }}
                      className="rounded"
                    />
                    <Icon className="h-4 w-4" />
                    <div>
                      <div className="text-sm font-medium">{label}</div>
                      <div className="text-xs text-muted-foreground">{description}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Export Actions */}
        <div className="flex items-center justify-between">
          <div className="flex space-x-2">
            <Button
              onClick={() => handleExport('csv')}
              disabled={!validation.canExport || isExporting}
              size="sm"
            >
              <FileText className="h-4 w-4 mr-1" />
              CSV
            </Button>
            
            <Button
              variant="outline"
              onClick={() => handleExport('excel')}
              disabled={!validation.canExport || isExporting}
              size="sm"
            >
              <FileSpreadsheet className="h-4 w-4 mr-1" />
              Excel
            </Button>
            
            <Button
              variant="outline"
              onClick={() => handleExport('json')}
              disabled={!validation.canExport || isExporting}
              size="sm"
            >
              <Database className="h-4 w-4 mr-1" />
              JSON
            </Button>
          </div>

          {showOptions && selectedFormats.length > 1 && (
            <Button
              onClick={handleBatchExport}
              disabled={!validation.canExport || isExporting}
              size="sm"
            >
              Export All ({selectedFormats.length})
            </Button>
          )}
        </div>

        {/* Export Jobs */}
        {exportJobs.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Recent Exports</h4>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {exportJobs.slice(0, 5).map((job) => (
                <div
                  key={job.id}
                  className="flex items-center justify-between p-2 bg-muted/30 rounded text-sm"
                >
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(job.status)}
                    <span className="font-mono text-xs">{job.filename}</span>
                    <Badge variant="outline" className="text-xs">
                      {job.format.toUpperCase()}
                    </Badge>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {job.status === 'processing' && job.progress !== undefined && (
                      <div className="w-16 h-1 bg-muted rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-primary transition-all duration-300"
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                    )}
                    
                    {job.status === 'error' && job.error && (
                      <span className="text-xs text-trading-loss">{job.error}</span>
                    )}
                    
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeJob(job.id)}
                      className="h-6 w-6 p-0"
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
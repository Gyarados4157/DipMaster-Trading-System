// 数据导出工具函数

/**
 * 将数据导出为CSV格式
 */
export function exportToCSV<T extends Record<string, any>>(
  data: T[],
  filename: string,
  columns?: { key: keyof T; label: string; formatter?: (value: any) => string }[]
): void {
  if (!data || data.length === 0) {
    console.warn('No data to export');
    return;
  }

  // 如果没有指定列，使用数据的所有键
  const exportColumns = columns || Object.keys(data[0]).map(key => ({
    key: key as keyof T,
    label: key,
    formatter: (value: any) => String(value)
  }));

  // 生成CSV头部
  const headers = exportColumns.map(col => col.label).join(',');
  
  // 生成CSV行
  const rows = data.map(row => 
    exportColumns.map(col => {
      const value = row[col.key];
      const formatted = col.formatter ? col.formatter(value) : String(value);
      
      // 处理包含逗号或引号的值
      if (formatted.includes(',') || formatted.includes('"') || formatted.includes('\n')) {
        return `"${formatted.replace(/"/g, '""')}"`;
      }
      return formatted;
    }).join(',')
  );

  // 组合CSV内容
  const csvContent = [headers, ...rows].join('\n');
  
  // 下载文件
  downloadFile(csvContent, filename, 'text/csv;charset=utf-8;');
}

/**
 * 将数据导出为JSON格式
 */
export function exportToJSON<T>(
  data: T,
  filename: string,
  prettyPrint: boolean = true
): void {
  const jsonContent = prettyPrint 
    ? JSON.stringify(data, null, 2)
    : JSON.stringify(data);
  
  downloadFile(jsonContent, filename, 'application/json;charset=utf-8;');
}

/**
 * 将数据导出为Excel格式（简单的XML格式）
 */
export function exportToExcel<T extends Record<string, any>>(
  data: T[],
  filename: string,
  sheetName: string = 'Sheet1',
  columns?: { key: keyof T; label: string; formatter?: (value: any) => string }[]
): void {
  if (!data || data.length === 0) {
    console.warn('No data to export');
    return;
  }

  const exportColumns = columns || Object.keys(data[0]).map(key => ({
    key: key as keyof T,
    label: key,
    formatter: (value: any) => String(value)
  }));

  // XML头部
  const xmlHeader = `<?xml version="1.0"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:x="urn:schemas-microsoft-com:office:excel"
 xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:html="http://www.w3.org/TR/REC-html40">
<Worksheet ss:Name="${sheetName}">
<Table>`;

  // 生成标题行
  const headerRow = `<Row>${exportColumns.map(col => `<Cell><Data ss:Type="String">${escapeXML(col.label)}</Data></Cell>`).join('')}</Row>`;

  // 生成数据行
  const dataRows = data.map(row => {
    const cells = exportColumns.map(col => {
      const value = row[col.key];
      const formatted = col.formatter ? col.formatter(value) : String(value);
      
      // 检测数据类型
      let cellType = 'String';
      if (typeof value === 'number') {
        cellType = 'Number';
      } else if (value instanceof Date) {
        cellType = 'DateTime';
      }
      
      return `<Cell><Data ss:Type="${cellType}">${escapeXML(formatted)}</Data></Cell>`;
    }).join('');
    
    return `<Row>${cells}</Row>`;
  }).join('');

  // XML尾部
  const xmlFooter = `</Table></Worksheet></Workbook>`;

  const xmlContent = xmlHeader + headerRow + dataRows + xmlFooter;
  
  downloadFile(xmlContent, filename, 'application/vnd.ms-excel;charset=utf-8;');
}

/**
 * 通用文件下载函数
 */
function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob(['\ufeff', content], { type: mimeType });
  const url = window.URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.style.display = 'none';
  
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  // 清理URL对象
  window.URL.revokeObjectURL(url);
}

/**
 * XML转义函数
 */
function escapeXML(str: string): string {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

/**
 * 格式化函数库
 */
export const formatters = {
  currency: (value: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  },
  
  percentage: (value: number, decimals: number = 2) => {
    return `${(value * 100).toFixed(decimals)}%`;
  },
  
  datetime: (value: string | Date) => {
    const date = typeof value === 'string' ? new Date(value) : value;
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  },
  
  date: (value: string | Date) => {
    const date = typeof value === 'string' ? new Date(value) : value;
    return date.toLocaleDateString('en-US');
  },
  
  time: (value: string | Date) => {
    const date = typeof value === 'string' ? new Date(value) : value;
    return date.toLocaleTimeString('en-US');
  },
  
  number: (value: number, decimals: number = 2) => {
    return value.toFixed(decimals);
  },
  
  largeNumber: (value: number) => {
    return new Intl.NumberFormat('en-US', {
      notation: 'compact',
      maximumFractionDigits: 2,
    }).format(value);
  },
};

/**
 * 预定义的导出配置
 */
export const exportConfigs = {
  trades: [
    { key: 'id', label: 'Trade ID' },
    { key: 'symbol', label: 'Symbol' },
    { key: 'side', label: 'Side' },
    { key: 'quantity', label: 'Quantity', formatter: (v: number) => formatters.number(v, 6) },
    { key: 'price', label: 'Price', formatter: (v: number) => formatters.currency(v) },
    { key: 'value', label: 'Value', formatter: (v: number) => formatters.currency(v) },
    { key: 'fee', label: 'Fee', formatter: (v: number) => formatters.currency(v) },
    { key: 'timestamp', label: 'Timestamp', formatter: formatters.datetime },
    { key: 'status', label: 'Status' },
  ],
  
  positions: [
    { key: 'symbol', label: 'Symbol' },
    { key: 'side', label: 'Side' },
    { key: 'quantity', label: 'Quantity', formatter: (v: number) => formatters.number(v, 6) },
    { key: 'entryPrice', label: 'Entry Price', formatter: (v: number) => formatters.currency(v) },
    { key: 'currentPrice', label: 'Current Price', formatter: (v: number) => formatters.currency(v) },
    { key: 'unrealizedPnl', label: 'Unrealized PnL', formatter: (v: number) => formatters.currency(v) },
    { key: 'realizedPnl', label: 'Realized PnL', formatter: (v: number) => formatters.currency(v) },
    { key: 'percentage', label: 'P&L %', formatter: (v: number) => formatters.percentage(v / 100) },
    { key: 'holdingTimeMinutes', label: 'Holding Time (min)' },
    { key: 'createdAt', label: 'Created At', formatter: formatters.datetime },
  ],
  
  pnl: [
    { key: 'timestamp', label: 'Timestamp', formatter: formatters.datetime },
    { key: 'totalPnl', label: 'Total PnL', formatter: (v: number) => formatters.currency(v) },
    { key: 'realizedPnl', label: 'Realized PnL', formatter: (v: number) => formatters.currency(v) },
    { key: 'unrealizedPnl', label: 'Unrealized PnL', formatter: (v: number) => formatters.currency(v) },
    { key: 'balance', label: 'Balance', formatter: (v: number) => formatters.currency(v) },
    { key: 'equity', label: 'Equity', formatter: (v: number) => formatters.currency(v) },
  ],
  
  alerts: [
    { key: 'id', label: 'Alert ID' },
    { key: 'type', label: 'Type' },
    { key: 'title', label: 'Title' },
    { key: 'message', label: 'Message' },
    { key: 'category', label: 'Category' },
    { key: 'source', label: 'Source' },
    { key: 'timestamp', label: 'Timestamp', formatter: formatters.datetime },
    { key: 'acknowledged', label: 'Acknowledged', formatter: (v: boolean) => v ? 'Yes' : 'No' },
    { key: 'acknowledgedBy', label: 'Acknowledged By' },
  ],
  
  riskMetrics: [
    { key: 'timestamp', label: 'Timestamp', formatter: formatters.datetime },
    { key: 'beta', label: 'Beta', formatter: (v: number) => formatters.number(v, 3) },
    { key: 'volatility', label: 'Volatility', formatter: (v: number) => formatters.percentage(v) },
    { key: 'expectedShortfall', label: 'Expected Shortfall', formatter: (v: number) => formatters.percentage(v) },
    { key: 'maxDrawdown', label: 'Max Drawdown', formatter: (v: number) => formatters.percentage(v) },
    { key: 'leverage', label: 'Leverage', formatter: (v: number) => `${formatters.number(v, 2)}x` },
  ],
};

/**
 * 生成带时间戳的文件名
 */
export function generateFilename(baseName: string, extension: string): string {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  return `${baseName}_${timestamp}.${extension}`;
}

/**
 * 批量导出多个数据集
 */
export function exportMultipleDatasets(
  datasets: Array<{
    data: any[];
    name: string;
    config?: any[];
  }>,
  format: 'csv' | 'json' | 'excel' = 'csv'
): void {
  datasets.forEach(({ data, name, config }) => {
    const filename = generateFilename(name, format);
    
    switch (format) {
      case 'csv':
        exportToCSV(data, filename, config);
        break;
      case 'json':
        exportToJSON(data, filename);
        break;
      case 'excel':
        exportToExcel(data, filename, name, config);
        break;
    }
  });
}

/**
 * 检查导出权限和数据量
 */
export function validateExport(data: any[], maxRows: number = 50000): { 
  canExport: boolean; 
  warning?: string; 
} {
  if (!data || data.length === 0) {
    return { canExport: false, warning: 'No data available to export' };
  }
  
  if (data.length > maxRows) {
    return { 
      canExport: true, 
      warning: `Large dataset (${data.length} rows). Export may take some time.` 
    };
  }
  
  return { canExport: true };
}
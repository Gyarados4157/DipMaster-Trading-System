'use client';

import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // 默认5分钟缓存
            staleTime: 1000 * 60 * 5,
            // 失败重试1次
            retry: 1,
            // 窗口聚焦时重新获取数据
            refetchOnWindowFocus: true,
            // 网络重连时重新获取数据
            refetchOnReconnect: true,
          },
          mutations: {
            // 突变失败重试1次
            retry: 1,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {process.env.NODE_ENV === 'development' && (
        <ReactQueryDevtools 
          initialIsOpen={false} 
          position="bottom-right"
        />
      )}
    </QueryClientProvider>
  );
}
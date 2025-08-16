import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { Providers } from '@/components/providers';
import { ThemeProvider } from '@/components/theme-provider';
import { Toaster } from 'react-hot-toast';

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({ 
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'DipMaster Trading System',
  description: 'Professional cryptocurrency trading dashboard with 82.1% win rate strategy',
  keywords: 'cryptocurrency, trading, DipMaster, dashboard, real-time, analytics',
  authors: [{ name: 'DipMaster Team' }],
  creator: 'DipMaster Trading System',
  publisher: 'DipMaster',
  robots: 'noindex, nofollow', // 私有系统，不允许搜索引擎索引
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
  },
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0f172a' },
  ],
  manifest: '/manifest.json',
  icons: {
    icon: '/favicon.ico',
    shortcut: '/favicon-16x16.png',
    apple: '/apple-touch-icon.png',
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://dipmaster.local',
    title: 'DipMaster Trading System',
    description: 'Professional cryptocurrency trading dashboard',
    siteName: 'DipMaster',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'DipMaster Trading System',
    description: 'Professional cryptocurrency trading dashboard',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`} suppressHydrationWarning>
      <head>
        <meta charSet="utf-8" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link rel="dns-prefetch" href="//api.binance.com" />
        <link rel="dns-prefetch" href="//stream.binance.com" />
      </head>
      <body className={`${inter.className} antialiased min-h-screen bg-background`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <Providers>
            <div className="relative flex min-h-screen flex-col">
              <main className="flex-1">
                {children}
              </main>
            </div>
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: 'hsl(var(--card))',
                  color: 'hsl(var(--card-foreground))',
                  border: '1px solid hsl(var(--border))',
                },
                success: {
                  iconTheme: {
                    primary: 'hsl(var(--primary))',
                    secondary: 'hsl(var(--primary-foreground))',
                  },
                },
                error: {
                  iconTheme: {
                    primary: 'hsl(var(--destructive))',
                    secondary: 'hsl(var(--destructive-foreground))',
                  },
                },
              }}
            />
          </Providers>
        </ThemeProvider>
      </body>
    </html>
  );
}
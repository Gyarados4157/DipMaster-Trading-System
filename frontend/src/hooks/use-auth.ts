import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import Cookies from 'js-cookie';
import { User, AuthState } from '@/types';

interface AuthStore extends AuthState {
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
  setUser: (user: User) => void;
  setToken: (token: string) => void;
  clearAuth: () => void;
}

const DEMO_USER: User = {
  id: 'demo-user-1',
  username: 'admin',
  email: 'admin@dipmaster.com',
  role: 'admin',
  permissions: ['read', 'write', 'admin'],
  lastLoginAt: new Date().toISOString(),
  isActive: true,
};

const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,

      login: async (username: string, password: string) => {
        set({ isLoading: true });

        try {
          // 模拟API调用延迟
          await new Promise(resolve => setTimeout(resolve, 1000));

          // Demo认证逻辑
          if (username === 'admin' && password === 'dipmaster123') {
            const token = `demo-token-${Date.now()}`;
            
            // 设置Cookie
            Cookies.set('auth-token', token, { 
              expires: 7, // 7天过期
              secure: process.env.NODE_ENV === 'production',
              sameSite: 'strict'
            });

            set({
              user: DEMO_USER,
              token,
              isAuthenticated: true,
              isLoading: false,
            });
          } else {
            throw new Error('Invalid credentials');
          }
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      logout: () => {
        // 清除Cookie
        Cookies.remove('auth-token');
        
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
        });
      },

      refreshToken: async () => {
        const currentToken = get().token;
        if (!currentToken) {
          throw new Error('No token to refresh');
        }

        try {
          // 在实际应用中，这里会调用刷新token的API
          // 现在只是模拟
          const newToken = `refreshed-token-${Date.now()}`;
          
          Cookies.set('auth-token', newToken, { 
            expires: 7,
            secure: process.env.NODE_ENV === 'production',
            sameSite: 'strict'
          });

          set({ token: newToken });
        } catch (error) {
          // 刷新失败，清除认证状态
          get().logout();
          throw error;
        }
      },

      setUser: (user: User) => {
        set({ user, isAuthenticated: true });
      },

      setToken: (token: string) => {
        Cookies.set('auth-token', token, { 
          expires: 7,
          secure: process.env.NODE_ENV === 'production',
          sameSite: 'strict'
        });
        set({ token });
      },

      clearAuth: () => {
        Cookies.remove('auth-token');
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
        });
      },
    }),
    {
      name: 'dipmaster-auth',
      storage: createJSONStorage(() => {
        // 使用sessionStorage以提高安全性
        return typeof window !== 'undefined' ? sessionStorage : {
          getItem: () => null,
          setItem: () => {},
          removeItem: () => {},
        };
      }),
      // 只持久化user信息，不持久化token
      partialize: (state) => ({ 
        user: state.user,
        isAuthenticated: state.isAuthenticated 
      }),
    }
  )
);

// 初始化认证状态
if (typeof window !== 'undefined') {
  const token = Cookies.get('auth-token');
  if (token) {
    useAuthStore.setState({ 
      token,
      isAuthenticated: true 
    });
  }
}

export const useAuth = () => {
  const store = useAuthStore();
  
  return {
    ...store,
    // 添加便捷方法
    hasPermission: (permission: string) => {
      return store.user?.permissions.includes(permission) ?? false;
    },
    hasRole: (role: string) => {
      return store.user?.role === role;
    },
    isAdmin: () => {
      return store.user?.role === 'admin';
    },
  };
};
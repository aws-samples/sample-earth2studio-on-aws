import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import {
  signIn as cognitoSignIn,
  signOut as cognitoSignOut,
  isAuthenticated as checkAuth,
  getIdToken,
} from "./cognito";
import { DEV_MODE } from "../config/auth";

// Self-signup is disabled at the Cognito User Pool level. The auth context
// only exposes signIn / signOut — administrators provision new users via
// `aws cognito-idp admin-create-user`.

interface AuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
  userEmail: string | null;
  error: string | null;
}

interface AuthContextType extends AuthState {
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => void;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function useAuth(): AuthContextType {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>({
    isAuthenticated: false,
    isLoading: true,
    userEmail: null,
    error: null,
  });

  useEffect(() => {
    if (DEV_MODE) {
      setState({ isAuthenticated: true, isLoading: false, userEmail: "dev@localhost", error: null });
      return;
    }
    checkAuth()
      .then(async (authed) => {
        if (authed) {
          const token = await getIdToken();
          const payload = JSON.parse(atob(token.split(".")[1]));
          setState({ isAuthenticated: true, isLoading: false, userEmail: payload.email || null, error: null });
        } else {
          setState((s) => ({ ...s, isLoading: false }));
        }
      })
      .catch(() => setState((s) => ({ ...s, isLoading: false })));
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    setState((s) => ({ ...s, error: null, isLoading: true }));
    try {
      await cognitoSignIn(email, password);
      setState({ isAuthenticated: true, isLoading: false, userEmail: email, error: null });
    } catch (err) {
      setState((s) => ({ ...s, isLoading: false, error: err instanceof Error ? err.message : "Sign in failed" }));
      throw err;
    }
  }, []);

  const signOut = useCallback(() => {
    cognitoSignOut();
    setState({ isAuthenticated: false, isLoading: false, userEmail: null, error: null });
  }, []);

  const clearError = useCallback(() => setState((s) => ({ ...s, error: null })), []);

  return (
    <AuthContext.Provider value={{ ...state, signIn, signOut, clearError }}>
      {children}
    </AuthContext.Provider>
  );
}

/**
 * Cognito authentication service using amazon-cognito-identity-js.
 * Sign-in only — self-signup is disabled at the User Pool level.
 * Administrators provision users via `aws cognito-idp admin-create-user`.
 *
 * In DEV_MODE, all functions are no-ops (Cognito is not initialized).
 */

import {
  CognitoUserPool,
  CognitoUser,
  AuthenticationDetails,
  CognitoUserSession,
} from "amazon-cognito-identity-js";
import { COGNITO_CONFIG, DEV_MODE } from "../config/auth";

let _userPool: CognitoUserPool | null = null;

function getUserPool(): CognitoUserPool {
  if (!_userPool) {
    _userPool = new CognitoUserPool({
      UserPoolId: COGNITO_CONFIG.userPoolId,
      ClientId: COGNITO_CONFIG.userPoolClientId,
    });
  }
  return _userPool;
}

/** Get the currently authenticated user (or null) */
export function getCurrentUser(): CognitoUser | null {
  if (DEV_MODE) return null;
  try {
    return getUserPool().getCurrentUser();
  } catch {
    return null;
  }
}

/** Get a valid ID token for API calls (auto-refreshes if expired) */
export function getIdToken(): Promise<string> {
  if (DEV_MODE) return Promise.reject(new Error("DEV_MODE"));
  return new Promise((resolve, reject) => {
    const user = getCurrentUser();
    if (!user) return reject(new Error("No authenticated user"));
    user.getSession((err: Error | null, session: CognitoUserSession | null) => {
      if (err || !session || !session.isValid()) return reject(err || new Error("Invalid session"));
      resolve(session.getIdToken().getJwtToken());
    });
  });
}

/** Sign in with email and password */
export function signIn(email: string, password: string): Promise<CognitoUserSession> {
  return new Promise((resolve, reject) => {
    const user = new CognitoUser({ Username: email, Pool: getUserPool() });
    const authDetails = new AuthenticationDetails({ Username: email, Password: password });
    user.authenticateUser(authDetails, {
      onSuccess: (session) => resolve(session),
      onFailure: (err) => reject(err),
      newPasswordRequired: () => reject(new Error("NEW_PASSWORD_REQUIRED")),
    });
  });
}

/** Sign out the current user */
export function signOut(): void {
  if (DEV_MODE) return;
  getCurrentUser()?.signOut();
}

/** Check if there is a valid session */
export function isAuthenticated(): Promise<boolean> {
  if (DEV_MODE) return Promise.resolve(false);
  return getIdToken().then(() => true).catch(() => false);
}

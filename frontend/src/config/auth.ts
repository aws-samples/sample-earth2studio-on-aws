/**
 * Cognito configuration.
 *
 * After CDK deployment, replace these values with the actual outputs:
 *   - UserPoolId from `cdk deploy` output
 *   - UserPoolClientId from `cdk deploy` output
 *
 * For local development, set VITE_DEV_MODE=true in frontend/.env.local
 * to bypass Cognito authentication entirely.
 *
 * For production with Cognito:
 *   VITE_USER_POOL_ID=us-west-2_XXXXXXXXX
 *   VITE_USER_POOL_CLIENT_ID=XXXXXXXXXXXXXXXXXXXXXXXXXX
 */
export const COGNITO_CONFIG = {
  userPoolId: import.meta.env.VITE_USER_POOL_ID || "us-west-2_PLACEHOLDER",
  userPoolClientId:
    import.meta.env.VITE_USER_POOL_CLIENT_ID || "PLACEHOLDER_CLIENT_ID",
  region: "us-west-2",
};

/** When true, skip Cognito auth (local dev with Flask backend) */
export const DEV_MODE = import.meta.env.VITE_DEV_MODE === "true";

module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', 'node_modules', '.eslintrc.cjs', 'vite.config.js'],
  parserOptions: { ecmaVersion: 'latest', sourceType: 'module', ecmaFeatures: { jsx: true } },
  settings: { react: { version: '18.2' } },
  plugins: ['react-refresh'],
  rules: {
    'react/prop-types': 'off',
    // Apostrophes/quotes in JSX text render correctly; escaping them is style only.
    'react/no-unescaped-entities': 'off',
    'no-empty': ['error', { allowEmptyCatch: true }],
    // react-three-fiber passes these straight to three.js objects
    'react/no-unknown-property': ['error', { ignore: ['args', 'position', 'rotation', 'scale', 'quaternion', 'intensity',
      'angle', 'penumbra', 'transparent', 'wireframe', 'emissive', 'emissiveIntensity', 'object', 'attach'] }],
    'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
  },
};

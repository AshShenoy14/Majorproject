import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [
        react(),
        tailwindcss(),
    ],
    resolve: {
        alias: {
            'fp-ts/es6': 'fp-ts/lib',
            'pdbe-molstar-css': 'pdbe-molstar/build/pdbe-molstar.css',
            'pdbe-molstar': 'pdbe-molstar/lib/index.js'
        },
    },
    optimizeDeps: {
        include: ['three'],
        exclude: ['three/webgpu']
    }
})

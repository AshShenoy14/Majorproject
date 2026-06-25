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
            'pdbe-molstar': 'pdbe-molstar/lib/index.js',
            './styles/pdbe-molstar-dark.scss': 'pdbe-molstar/build/pdbe-molstar.css',
            './styles/pdbe-molstar-light.scss': 'pdbe-molstar/build/pdbe-molstar.css'
        },
    },
    optimizeDeps: {
        include: ['three'],
        exclude: ['three/webgpu']
    },
    css: {
        preprocessorOptions: {
            scss: {
                api: 'modern',
                loadPaths: ['node_modules']
            }
        }
    }
})

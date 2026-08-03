// vite.config.js
import { defineConfig } from "file:///E:/majorproject/app/frontend/node_modules/vite/dist/node/index.js";
import react from "file:///E:/majorproject/app/frontend/node_modules/@vitejs/plugin-react/dist/index.js";
import tailwindcss from "file:///E:/majorproject/app/frontend/node_modules/@tailwindcss/vite/dist/index.mjs";
var vite_config_default = defineConfig({
  plugins: [
    react(),
    tailwindcss()
  ],
  resolve: {
    alias: {
      "fp-ts/es6": "fp-ts/lib",
      "pdbe-molstar-css": "pdbe-molstar/build/pdbe-molstar.css",
      "pdbe-molstar": "pdbe-molstar/lib/index.js",
      "./styles/pdbe-molstar-dark.scss": "pdbe-molstar/build/pdbe-molstar.css",
      "./styles/pdbe-molstar-light.scss": "pdbe-molstar/build/pdbe-molstar.css"
    }
  },
  optimizeDeps: {
    include: ["three"],
    exclude: ["three/webgpu"]
  },
  css: {
    preprocessorOptions: {
      scss: {
        api: "modern",
        loadPaths: ["node_modules"]
      }
    }
  }
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcuanMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJFOlxcXFxtYWpvcnByb2plY3RcXFxcYXBwXFxcXGZyb250ZW5kXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ZpbGVuYW1lID0gXCJFOlxcXFxtYWpvcnByb2plY3RcXFxcYXBwXFxcXGZyb250ZW5kXFxcXHZpdGUuY29uZmlnLmpzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9FOi9tYWpvcnByb2plY3QvYXBwL2Zyb250ZW5kL3ZpdGUuY29uZmlnLmpzXCI7aW1wb3J0IHsgZGVmaW5lQ29uZmlnIH0gZnJvbSAndml0ZSdcclxuaW1wb3J0IHJlYWN0IGZyb20gJ0B2aXRlanMvcGx1Z2luLXJlYWN0J1xyXG5pbXBvcnQgdGFpbHdpbmRjc3MgZnJvbSAnQHRhaWx3aW5kY3NzL3ZpdGUnXHJcblxyXG5pbXBvcnQgcGF0aCBmcm9tICdwYXRoJ1xyXG5cclxuLy8gaHR0cHM6Ly92aXRlanMuZGV2L2NvbmZpZy9cclxuZXhwb3J0IGRlZmF1bHQgZGVmaW5lQ29uZmlnKHtcclxuICAgIHBsdWdpbnM6IFtcclxuICAgICAgICByZWFjdCgpLFxyXG4gICAgICAgIHRhaWx3aW5kY3NzKCksXHJcbiAgICBdLFxyXG4gICAgcmVzb2x2ZToge1xyXG4gICAgICAgIGFsaWFzOiB7XHJcbiAgICAgICAgICAgICdmcC10cy9lczYnOiAnZnAtdHMvbGliJyxcclxuICAgICAgICAgICAgJ3BkYmUtbW9sc3Rhci1jc3MnOiAncGRiZS1tb2xzdGFyL2J1aWxkL3BkYmUtbW9sc3Rhci5jc3MnLFxyXG4gICAgICAgICAgICAncGRiZS1tb2xzdGFyJzogJ3BkYmUtbW9sc3Rhci9saWIvaW5kZXguanMnLFxyXG4gICAgICAgICAgICAnLi9zdHlsZXMvcGRiZS1tb2xzdGFyLWRhcmsuc2Nzcyc6ICdwZGJlLW1vbHN0YXIvYnVpbGQvcGRiZS1tb2xzdGFyLmNzcycsXHJcbiAgICAgICAgICAgICcuL3N0eWxlcy9wZGJlLW1vbHN0YXItbGlnaHQuc2Nzcyc6ICdwZGJlLW1vbHN0YXIvYnVpbGQvcGRiZS1tb2xzdGFyLmNzcydcclxuICAgICAgICB9LFxyXG4gICAgfSxcclxuICAgIG9wdGltaXplRGVwczoge1xyXG4gICAgICAgIGluY2x1ZGU6IFsndGhyZWUnXSxcclxuICAgICAgICBleGNsdWRlOiBbJ3RocmVlL3dlYmdwdSddXHJcbiAgICB9LFxyXG4gICAgY3NzOiB7XHJcbiAgICAgICAgcHJlcHJvY2Vzc29yT3B0aW9uczoge1xyXG4gICAgICAgICAgICBzY3NzOiB7XHJcbiAgICAgICAgICAgICAgICBhcGk6ICdtb2Rlcm4nLFxyXG4gICAgICAgICAgICAgICAgbG9hZFBhdGhzOiBbJ25vZGVfbW9kdWxlcyddXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9XHJcbn0pXHJcbiJdLAogICJtYXBwaW5ncyI6ICI7QUFBOFEsU0FBUyxvQkFBb0I7QUFDM1MsT0FBTyxXQUFXO0FBQ2xCLE9BQU8saUJBQWlCO0FBS3hCLElBQU8sc0JBQVEsYUFBYTtBQUFBLEVBQ3hCLFNBQVM7QUFBQSxJQUNMLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxFQUNoQjtBQUFBLEVBQ0EsU0FBUztBQUFBLElBQ0wsT0FBTztBQUFBLE1BQ0gsYUFBYTtBQUFBLE1BQ2Isb0JBQW9CO0FBQUEsTUFDcEIsZ0JBQWdCO0FBQUEsTUFDaEIsbUNBQW1DO0FBQUEsTUFDbkMsb0NBQW9DO0FBQUEsSUFDeEM7QUFBQSxFQUNKO0FBQUEsRUFDQSxjQUFjO0FBQUEsSUFDVixTQUFTLENBQUMsT0FBTztBQUFBLElBQ2pCLFNBQVMsQ0FBQyxjQUFjO0FBQUEsRUFDNUI7QUFBQSxFQUNBLEtBQUs7QUFBQSxJQUNELHFCQUFxQjtBQUFBLE1BQ2pCLE1BQU07QUFBQSxRQUNGLEtBQUs7QUFBQSxRQUNMLFdBQVcsQ0FBQyxjQUFjO0FBQUEsTUFDOUI7QUFBQSxJQUNKO0FBQUEsRUFDSjtBQUNKLENBQUM7IiwKICAibmFtZXMiOiBbXQp9Cg==

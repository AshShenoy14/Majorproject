import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { PerspectiveCamera, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';

const Helix = ({ count = 40, radius = 2, height = 8 }) => {
  const points = useMemo(() => {
    const p = [];
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 4;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = (i / count) * height - height / 2;
      p.push(new THREE.Vector3(x, y, z));
    }
    return p;
  }, [count, radius, height]);

  const groupRef = useRef();

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.01;
      groupRef.current.rotation.z += 0.005;
    }
  });

  return (
    <group ref={groupRef}>
      {points.map((point, i) => (
        <group key={i} position={point}>
          <mesh>
            <sphereGeometry args={[0.2, 16, 16]} />
            <meshStandardMaterial
              color={i % 2 === 0 ? "#0D9488" : "#7C3AED"}
              emissive={i % 2 === 0 ? "#0D9488" : "#7C3AED"}
              emissiveIntensity={0.5}
            />
          </mesh>
          {i < points.length - 1 && (
            <mesh
              position={point.clone().lerp(points[i + 1], 0.5).sub(point)}
              quaternion={new THREE.Quaternion().setFromUnitVectors(
                new THREE.Vector3(0, 1, 0),
                points[i + 1].clone().sub(point).normalize()
              )}
            >
              <cylinderGeometry args={[0.05, 0.05, point.distanceTo(points[i + 1]), 8]} />
              <meshStandardMaterial color="#64748b" transparent opacity={0.3} />
            </mesh>
          )}
        </group>
      ))}
    </group>
  );
};

const ProteinPreloader = ({ progress, onComplete }) => {
  const [exit, setExit] = useState(false);

  useEffect(() => {
    if (progress >= 100) {
      const timer = setTimeout(() => {
        setExit(true);
        setTimeout(onComplete, 1000);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [progress, onComplete]);

  return (
    <AnimatePresence>
      {!exit && (
        <motion.div
          initial={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 1.2, ease: [0.4, 0, 0.2, 1] }}
          className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-[#020617] overflow-hidden"
        >
          {/* Ambient Background Gradient */}
          <div className="absolute inset-0 z-0 opacity-30">
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-teal-500/10 blur-[120px] rounded-full" />
            <div className="absolute top-1/4 left-1/4 w-[400px] h-[400px] bg-violet-500/5 blur-[100px] rounded-full" />
          </div>

          <div className="relative z-10 w-full flex-1 min-h-0">
            <Canvas dpr={[1, 2]}>
              <PerspectiveCamera makeDefault position={[0, 0, 12]} />
              <ambientLight intensity={0.6} />
              <pointLight position={[10, 10, 10]} intensity={1.5} />
              <spotLight position={[-10, 10, 10]} angle={0.15} penumbra={1} intensity={1.5} />
              <Helix />
              <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={1} />
            </Canvas>
          </div>

          <div className="relative z-10 flex flex-col items-center gap-6 flex-shrink-0 pb-20">
            <div className="text-center">
              <motion.h1
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="text-5xl font-black text-white tracking-tighter mb-2"
              >
                Trans<span className="text-teal-400">Graph</span>
              </motion.h1>
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.5 }}
                className="text-teal-100 text-[10px] font-black uppercase tracking-[0.4em]"
              >
                Neural Interactome Intelligence
              </motion.p>
            </div>

            <div className="flex flex-col items-center gap-3">
              <div className="w-80 h-1 bg-white/5 rounded-full overflow-hidden border border-white/5">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  className="h-full bg-gradient-to-r from-teal-400 via-teal-500 to-violet-500 shadow-[0_0_15px_rgba(45,212,191,0.4)]"
                />
              </div>

              <div className="flex items-center gap-4 w-full justify-between">
                <div className="text-[10px] text-teal-500/50 font-bold uppercase tracking-widest">
                  {progress < 40 ? 'Synthesizing...' : progress < 80 ? 'Optimizing GNN...' : 'Finalizing...'}
                </div>
                <div className="text-white font-black text-xs tabular-nums tracking-wider">
                  {Math.round(progress)}%
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default ProteinPreloader
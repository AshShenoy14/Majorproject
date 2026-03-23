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
          transition={{ duration: 0.8, ease: "easeInOut" }}
          className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-slate-950 overflow-hidden"
        >
          <div className="relative z-10 w-full flex-1 min-h-0">
            <Canvas dpr={[1, 2]}>
              <PerspectiveCamera makeDefault position={[0, 0, 14]} />
              <ambientLight intensity={0.5} />
              <pointLight position={[10, 10, 10]} intensity={1} />
              <spotLight position={[-10, 10, 10]} angle={0.15} penumbra={1} intensity={1} />
              <Helix />
              <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.5} />
            </Canvas>
          </div>

          <div className="relative z-10 flex flex-col items-center gap-4 flex-shrink-0 pb-10">
            <motion.h1
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-purple-400 tracking-tight"
            >
              TransGraph-PPI
            </motion.h1>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.6 }}
              className="text-slate-400 text-sm font-medium uppercase tracking-[0.2em]"
            >
              Initializing Neural Interactome
            </motion.p>

            <div className="mt-4 w-64 h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                className="h-full bg-gradient-to-r from-teal-500 to-purple-500 shadow-[0_0_10px_rgba(20,184,166,0.5)]"
              />
            </div>

            <div className="text-slate-500 font-mono text-xs tabular-nums mt-1">
              {Math.round(progress)}%
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default ProteinPreloader
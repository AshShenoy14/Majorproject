import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { PDB_1YCR_DATA } from './pdb1ycrData';

/**
 * Creates a smooth radial glow texture for interface bio-luminescence
 */
function createGlowTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  gradient.addColorStop(0, 'rgba(52, 211, 153, 0.95)');
  gradient.addColorStop(0.3, 'rgba(56, 189, 248, 0.6)');
  gradient.addColorStop(0.7, 'rgba(16, 185, 129, 0.15)');
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 128, 128);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

/**
 * Creates smooth CatmullRom backbone ribbon tube
 */
function createBackboneTube(caCoords, radius, color) {
  if (!caCoords || caCoords.length < 3) return null;
  const points = caCoords.map(([x, y, z]) => new THREE.Vector3(x, y, z));
  const curve = new THREE.CatmullRomCurve3(points);
  curve.curveType = 'centripetal';
  curve.tension = 0.4;

  const segments = Math.max(caCoords.length * 5, 60);
  const geometry = new THREE.TubeGeometry(curve, segments, radius, 12, false);
  const material = new THREE.MeshPhysicalMaterial({
    color: color,
    roughness: 0.3,
    metalness: 0.1,
    clearcoat: 0.6,
    clearcoatRoughness: 0.2,
    reflectivity: 0.5,
  });

  const mesh = new THREE.Mesh(geometry, material);
  return { mesh, material };
}

/**
 * Creates instanced molecular surface / atom spheres
 */
function createAtomSpheres(atoms, baseColor, radiusScale = 0.82) {
  const count = atoms.length;
  const sphereGeo = new THREE.SphereGeometry(1, 16, 14);
  const material = new THREE.MeshPhysicalMaterial({
    color: baseColor,
    roughness: 0.38,
    metalness: 0.05,
    clearcoat: 0.45,
    clearcoatRoughness: 0.25,
  });

  const instancedMesh = new THREE.InstancedMesh(sphereGeo, material, count);
  const dummy = new THREE.Object3D();
  const cColor = new THREE.Color();
  const baseCol = new THREE.Color(baseColor);

  // Element radius approximate mapping (scaled for molecular surface)
  const radii = {
    C: 1.5 * radiusScale,
    N: 1.4 * radiusScale,
    O: 1.35 * radiusScale,
    S: 1.7 * radiusScale,
    P: 1.7 * radiusScale,
  };

  for (let i = 0; i < count; i++) {
    const [x, y, z, elem, resName, resSeq, atomName] = atoms[i];
    const r = radii[elem] || 1.45 * radiusScale;

    dummy.position.set(x, y, z);
    dummy.scale.set(r, r, r);
    dummy.updateMatrix();
    instancedMesh.setMatrixAt(i, dummy.matrix);

    // Subtle natural molecular tint variations for depth
    cColor.copy(baseCol);
    if (elem === 'O') {
      cColor.offsetHSL(-0.03, 0.08, 0.06);
    } else if (elem === 'N') {
      cColor.offsetHSL(0.04, 0.06, 0.08);
    } else if (elem === 'S') {
      cColor.offsetHSL(0.08, 0.15, 0.1);
    } else if (atomName === 'CA') {
      cColor.offsetHSL(0, 0, -0.04);
    }
    instancedMesh.setColorAt(i, cColor);
  }

  instancedMesh.instanceMatrix.needsUpdate = true;
  if (instancedMesh.instanceColor) {
    instancedMesh.instanceColor.needsUpdate = true;
  }

  return { instancedMesh, material };
}

const HeroProtein3D = () => {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);

  // Hover tooltip state
  const [tooltip, setTooltip] = useState(null);
  const [isInteracting, setIsInteracting] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    let width = container.clientWidth || 480;
    let height = container.clientHeight || 450;

    // 1. Scene setup
    const scene = new THREE.Scene();

    // 2. Camera setup
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 1000);
    camera.position.set(0, 0, 72);

    // 3. Renderer setup
    const renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;

    // 4. Lighting setup (high-end scientific studio lighting)
    const ambientLight = new THREE.AmbientLight(0xf8fafc, 1.3);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 2.2);
    dirLight1.position.set(30, 40, 45);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 1.2);
    dirLight2.position.set(-35, -25, -20);
    scene.add(dirLight2);

    const rimLight = new THREE.DirectionalLight(0x34d399, 1.0);
    rimLight.position.set(20, -30, -35);
    scene.add(rimLight);

    // 5. Molecular Complex Group
    const complexGroup = new THREE.Group();
    scene.add(complexGroup);

    // Initial slight orientation tilt for optimal binding site perspective
    complexGroup.rotation.x = 0.22;
    complexGroup.rotation.y = -0.45;

    // Colors: Distinct, elegant scientific soft green and blue
    const mdm2Color = 0x38bdf8; // Soft scientific cyan/blue
    const tp53Color = 0x10b981; // Soft scientific emerald green

    // MDM2 (Chain A) meshes
    const mdm2Backbone = createBackboneTube(PDB_1YCR_DATA.mdm2.ca, 0.72, mdm2Color);
    const mdm2Atoms = createAtomSpheres(PDB_1YCR_DATA.mdm2.atoms, mdm2Color, 0.8);
    mdm2Backbone.mesh.userData = { target: 'mdm2' };
    mdm2Atoms.instancedMesh.userData = { target: 'mdm2' };

    complexGroup.add(mdm2Backbone.mesh);
    complexGroup.add(mdm2Atoms.instancedMesh);

    // TP53 (Chain B) meshes
    const tp53Backbone = createBackboneTube(PDB_1YCR_DATA.tp53.ca, 0.85, tp53Color);
    const tp53Atoms = createAtomSpheres(PDB_1YCR_DATA.tp53.atoms, tp53Color, 0.88);
    tp53Backbone.mesh.userData = { target: 'tp53' };
    tp53Atoms.instancedMesh.userData = { target: 'tp53' };

    complexGroup.add(tp53Backbone.mesh);
    complexGroup.add(tp53Atoms.instancedMesh);

    // 6. Subtle glowing highlight at the protein-protein interaction interface
    const ifCenter = PDB_1YCR_DATA.interface.center;
    const interfacePos = new THREE.Vector3(ifCenter[0], ifCenter[1], ifCenter[2]);

    const glowTexture = createGlowTexture();
    const glowMat = new THREE.SpriteMaterial({
      map: glowTexture,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.62,
      depthWrite: false,
    });
    const interfaceGlow = new THREE.Sprite(glowMat);
    interfaceGlow.position.copy(interfacePos);
    interfaceGlow.scale.set(22, 22, 1);
    interfaceGlow.userData = { target: 'interface' };
    complexGroup.add(interfaceGlow);

    // Secondary subtle contact halo
    const innerGlowMat = new THREE.SpriteMaterial({
      map: glowTexture,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.45,
      depthWrite: false,
    });
    const innerGlow = new THREE.Sprite(innerGlowMat);
    innerGlow.position.copy(interfacePos);
    innerGlow.scale.set(12, 12, 1);
    complexGroup.add(innerGlow);

    // Interactive hit sphere for interface detection
    const ifHitGeo = new THREE.SphereGeometry(6.5, 12, 12);
    const ifHitMat = new THREE.MeshBasicMaterial({ visible: false });
    const ifHitMesh = new THREE.Mesh(ifHitGeo, ifHitMat);
    ifHitMesh.position.copy(interfacePos);
    ifHitMesh.userData = { target: 'interface' };
    complexGroup.add(ifHitMesh);

    // Delicate contact dotted lines between the key anchor residues
    const contactLineGroup = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x6ee7b7,
      transparent: true,
      opacity: 0.5,
      blending: THREE.AdditiveBlending,
    });

    PDB_1YCR_DATA.interface.contacts.forEach((contact) => {
      const lineGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...contact.from),
        new THREE.Vector3(...contact.to),
      ]);
      const line = new THREE.Line(lineGeo, lineMat);
      contactLineGroup.add(line);
    });
    complexGroup.add(contactLineGroup);

    // 7. Raycasting & Mouse Interaction Setup
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(-999, -999);
    const interactiveObjects = [
      mdm2Atoms.instancedMesh,
      mdm2Backbone.mesh,
      tp53Atoms.instancedMesh,
      tp53Backbone.mesh,
      ifHitMesh,
    ];

    // Drag rotation physics state
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;
    let velX = 0;
    let velY = 0;
    const friction = 0.93;
    const baseIdleSpeedY = 0.0035;

    const onPointerDown = (e) => {
      isDragging = true;
      setIsInteracting(true);
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
      velX = 0;
      velY = 0;
      if (canvasRef.current) {
        canvasRef.current.style.cursor = 'grabbing';
      }
    };

    const onPointerMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const clientX = e.clientX;
      const clientY = e.clientY;

      // Update normalized mouse coordinates for raycasting
      mouse.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((clientY - rect.top) / rect.height) * 2 + 1;

      if (isDragging) {
        const deltaX = clientX - prevMouseX;
        const deltaY = clientY - prevMouseY;
        prevMouseX = clientX;
        prevMouseY = clientY;

        velY = deltaX * 0.0065;
        velX = deltaY * 0.0065;

        complexGroup.rotation.y += velY;
        complexGroup.rotation.x += velX;
      }
    };

    const onPointerUp = () => {
      if (isDragging) {
        isDragging = false;
        setIsInteracting(false);
        if (canvasRef.current) {
          canvasRef.current.style.cursor = 'grab';
        }
      }
    };

    const onPointerLeave = () => {
      isDragging = false;
      setIsInteracting(false);
      mouse.x = -999;
      mouse.y = -999;
      setTooltip(null);
      if (canvasRef.current) {
        canvasRef.current.style.cursor = 'grab';
      }
    };

    canvas.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointerleave', onPointerLeave);
    canvas.style.cursor = 'grab';

    // 8. Animation & Render Loop
    let animationFrameId;
    let clock = new THREE.Clock();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const delta = clock.getDelta();
      const elapsedTime = clock.getElapsedTime();

      // Continuous, smooth 360° idle rotation with graceful momentum blending
      if (!isDragging) {
        // Apply inertia velocity decay
        velX *= friction;
        velY *= friction;

        complexGroup.rotation.x += velX;
        complexGroup.rotation.y += velY + baseIdleSpeedY;

        // Subtle gentle natural breathing tilt oscillation
        complexGroup.position.y = Math.sin(elapsedTime * 0.8) * 0.4;
      }

      // Subtle biological pulsation of the binding interface glow
      const pulse = 0.55 + 0.18 * Math.sin(elapsedTime * 2.2);
      glowMat.opacity = pulse;
      innerGlowMat.opacity = pulse * 0.7;
      const glowScale = 21 + 2.5 * Math.sin(elapsedTime * 2.2);
      interfaceGlow.scale.set(glowScale, glowScale, 1);

      // Raycasting for subtle hover response and minimal tooltip
      if (!isDragging && mouse.x > -2 && mouse.y > -2) {
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(interactiveObjects, false);

        if (intersects.length > 0) {
          const hit = intersects[0];
          const target = hit.object.userData?.target;

          if (target === 'tp53') {
            tp53Atoms.material.emissive.setHex(0x059669);
            tp53Atoms.material.emissiveIntensity = 0.28;
            tp53Backbone.material.emissive.setHex(0x059669);
            tp53Backbone.material.emissiveIntensity = 0.35;

            mdm2Atoms.material.emissiveIntensity = 0;
            mdm2Backbone.material.emissiveIntensity = 0;

            setTooltip({
              name: 'TP53',
              badge: 'Chain B',
              role: 'Tumor Suppressor p53',
              detail: 'Transactivation α-helix docked into MDM2 binding cleft',
              color: 'text-emerald-500',
              bgDot: 'bg-emerald-400',
            });
          } else if (target === 'mdm2') {
            mdm2Atoms.material.emissive.setHex(0x0284c7);
            mdm2Atoms.material.emissiveIntensity = 0.26;
            mdm2Backbone.material.emissive.setHex(0x0284c7);
            mdm2Backbone.material.emissiveIntensity = 0.32;

            tp53Atoms.material.emissiveIntensity = 0;
            tp53Backbone.material.emissiveIntensity = 0;

            setTooltip({
              name: 'MDM2',
              badge: 'Chain A',
              role: 'E3 Ubiquitin Ligase',
              detail: 'Hydrophobic cleft domain interacting with TP53 peptide',
              color: 'text-sky-500',
              bgDot: 'bg-sky-400',
            });
          } else if (target === 'interface') {
            glowMat.opacity = 0.9;
            innerGlowMat.opacity = 0.85;

            setTooltip({
              name: 'Interaction Interface',
              badge: 'Binding Pocket',
              role: 'PPI Interface Region',
              detail: 'Phe19, Trp23, Leu26 triad inserting into hydrophobic cleft',
              color: 'text-teal-400',
              bgDot: 'bg-teal-300 animate-ping',
            });
          }
        } else {
          // Reset highlights
          tp53Atoms.material.emissiveIntensity = 0;
          tp53Backbone.material.emissiveIntensity = 0;
          mdm2Atoms.material.emissiveIntensity = 0;
          mdm2Backbone.material.emissiveIntensity = 0;
          setTooltip(null);
        }
      }

      renderer.render(scene, camera);
    };

    animate();

    // 9. Resize handler
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      width = container.clientWidth;
      height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    window.addEventListener('resize', handleResize);

    // 10. Cleanup
    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      canvas.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointerleave', onPointerLeave);

      // Dispose Three resources
      glowTexture.dispose();
      glowMat.dispose();
      innerGlowMat.dispose();
      mdm2Atoms.material.dispose();
      tp53Atoms.material.dispose();
      mdm2Backbone.material.dispose();
      tp53Backbone.material.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full min-h-[380px] lg:min-h-[440px] flex items-center justify-center select-none"
    >
      <canvas
        ref={canvasRef}
        className="w-full h-full block focus:outline-none"
        style={{ touchAction: 'none' }}
      />

      {/* Small Minimal Hover Tooltip (Unobtrusive & Clean) */}
      {tooltip && (
        <div className="absolute top-4 right-4 pointer-events-none transition-all duration-200 ease-out z-30">
          <div className="backdrop-blur-md bg-slate-900/90 text-white px-3.5 py-2.5 rounded-xl shadow-xl border border-slate-700/60 max-w-[260px]">
            <div className="flex items-center gap-2 mb-1">
              <span className={`w-2 h-2 rounded-full ${tooltip.bgDot}`} />
              <span className="font-bold text-xs tracking-wide text-slate-100">
                {tooltip.name}
              </span>
              <span className="text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 ml-auto border border-slate-700">
                {tooltip.badge}
              </span>
            </div>
            <div className="text-[11px] font-medium text-slate-300 leading-snug">
              {tooltip.role}
            </div>
            <div className="text-[10px] text-slate-400 mt-1 leading-tight border-t border-slate-800/80 pt-1 font-light">
              {tooltip.detail}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HeroProtein3D;

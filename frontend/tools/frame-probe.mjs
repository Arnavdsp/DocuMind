import { chromium } from "playwright";
import { readFileSync } from "node:fs";
const frag = readFileSync("src/sim/schwarzschild.frag.glsl","utf8");
const vert = readFileSync("src/sim/schwarzschild.vert.glsl","utf8");
const W=800,H=450;
function camera(d,inc,az){const p=[d*Math.cos(inc)*Math.cos(az),d*Math.sin(inc),d*Math.cos(inc)*Math.sin(az)];
 const l=Math.hypot(...p)||1;const f=p.map(v=>v/l);let up=Math.abs(f[1])>0.999?[0,0,1]:[0,1,0];
 let r=[up[1]*f[2]-up[2]*f[1],up[2]*f[0]-up[0]*f[2],up[0]*f[1]-up[1]*f[0]];const rl=Math.hypot(...r)||1;r=r.map(v=>v/rl);
 const t=[f[1]*r[2]-f[2]*r[1],f[2]*r[0]-f[0]*r[2],f[0]*r[1]-f[1]*r[0]];return {pos:p,basis:[...r,...t,...f]};}

const RS = 1.13, OUTER_RS = 9.14;
const trials = [];
for (const st of [110,180,300,460]) for (const k of [2.0,2.6,3.2]) trials.push({k,steps:st,distRs:k*OUTER_RS});

const b = await chromium.launch({executablePath:"/opt/pw-browsers/chromium",
  args:["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"]});
const p = await b.newPage({viewport:{width:W,height:H}});
await p.setContent(`<canvas id=c width=${W} height=${H}></canvas>`);
const res = await p.evaluate(({vert,frag,trials,W,H,RS,OUTER_RS})=>{
  const gl=document.getElementById("c").getContext("webgl",{preserveDrawingBuffer:true});
  const mk=(t,s)=>{const sh=gl.createShader(t);gl.shaderSource(sh,s);gl.compileShader(sh);
    if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(sh));return sh;};
  const pr=gl.createProgram();gl.attachShader(pr,mk(gl.VERTEX_SHADER,vert));gl.attachShader(pr,mk(gl.FRAGMENT_SHADER,frag));
  gl.linkProgram(pr);gl.useProgram(pr);
  const bf=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,bf);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  const lo=gl.getAttribLocation(pr,"aPosition");gl.enableVertexAttribArray(lo);gl.vertexAttribPointer(lo,2,gl.FLOAT,false,0,0);
  gl.viewport(0,0,W,H);
  const U=n=>gl.getUniformLocation(pr,n);
  const out=[];
  for(const t of trials){
    const d=t.distRs*RS;
    const inc=0.29, az=0;
    const pos=[d*Math.cos(inc)*Math.cos(az), d*Math.sin(inc), d*Math.cos(inc)*Math.sin(az)];
    const l=Math.hypot(...pos);const f=pos.map(v=>v/l);
    let up=[0,1,0];
    let r=[up[1]*f[2]-up[2]*f[1],up[2]*f[0]-up[0]*f[2],up[0]*f[1]-up[1]*f[0]];
    const rl=Math.hypot(...r);r=r.map(v=>v/rl);
    const tt=[f[1]*r[2]-f[2]*r[1],f[2]*r[0]-f[0]*r[2],f[0]*r[1]-f[1]*r[0]];
    gl.uniform2f(U("uResolution"),W,H);gl.uniform1f(U("uTime"),3);
    gl.uniform3f(U("uCamPos"),...pos);
    gl.uniformMatrix3fv(U("uCamBasis"),false,new Float32Array([...r,...tt,...f]));
    gl.uniform1f(U("uFov"),0.9);
    gl.uniform1i(U("uSteps"),t.steps);gl.uniform1i(U("uDiskSamples"),2);
    gl.uniform1f(U("uRs"),RS);gl.uniform1f(U("uDiskInner"),RS*3);
    gl.uniform1f(U("uDiskOuter"),RS*OUTER_RS);
    gl.uniform1f(U("uDiskLuminosity"),0.9);gl.uniform1f(U("uDiskIntegrity"),1);
    gl.uniform1f(U("uParticleSeed"),9);gl.uniform1f(U("uCollapse"),1);
    gl.uniform1f(U("uGrounding"),1);gl.uniform1f(U("uAbstained"),0);
    gl.uniform1f(U("uTraceProgress"),1);gl.uniform1i(U("uHotspotCount"),0);
    gl.uniform3fv(U("uHotspots[0]"),new Float32Array(24));gl.uniform1f(U("uRedshift"),0);
    gl.drawArrays(gl.TRIANGLES,0,3);gl.finish();
    const px=new Uint8Array(W*H*4);gl.readPixels(0,0,W,H,gl.RGBA,gl.UNSIGNED_BYTE,px);
    let minX=W,maxX=0,minY=H,maxY=0,lit=0;
    for(let y=0;y<H;y++)for(let x=0;x<W;x++){const i=(y*W+x)*4;
      const L=(px[i]+px[i+1]+px[i+2])/3; const warm=px[i]>px[i+2]+18; if(L>40&&warm){lit++;if(x<minX)minX=x;if(x>maxX)maxX=x;if(y<minY)minY=y;if(y>maxY)maxY=y;}}
    out.push({k:t.k,steps:t.steps,distRs:+t.distRs.toFixed(1),
      widthPct:+(((maxX-minX)/W)*100).toFixed(1),heightPct:+(((maxY-minY)/H)*100).toFixed(1),
      litPct:+((lit/(W*H))*100).toFixed(1)});
  }
  return out;
},{vert,frag,trials,W,H,RS,OUTER_RS});
await b.close();
console.log("steps  k     dist(rs)  diskWidth%  diskArea%");
for(const r of res) console.log(`${String(r.steps).padEnd(6)} ${String(r.k).padEnd(5)} ${String(r.distRs).padStart(7)} ${String(r.widthPct).padStart(11)} ${String(r.litPct).padStart(10)}`);

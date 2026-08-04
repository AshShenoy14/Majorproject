import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bot,
  Send,
  Sparkles,
  User,
  Loader2,
  BookOpen,
  ExternalLink,
  ChevronRight,
  Dna,
  FlaskConical,
  HeartPulse,
  Pill,
  BrainCircuit
} from 'lucide-react';
import { ppiService } from '../services/api';

// Simple markdown-like renderer for bold, headers, code, and emoji
const renderMarkdown = (text) => {
  if (!text) return null;
  
  const lines = text.split('\n');
  const elements = [];
  
  lines.forEach((line, lineIdx) => {
    let content = line;
    
    // Headers
    if (content.startsWith('## ')) {
      elements.push(
        <h2 key={lineIdx} className="text-lg font-bold text-slate-800 mt-4 mb-2 flex items-center gap-2">
          {content.replace('## ', '')}
        </h2>
      );
      return;
    }
    if (content.startsWith('### ')) {
      elements.push(
        <h3 key={lineIdx} className="text-md font-bold text-slate-700 mt-3 mb-1">
          {content.replace('### ', '')}
        </h3>
      );
      return;
    }
    
    // Empty lines
    if (content.trim() === '') {
      elements.push(<div key={lineIdx} className="h-2" />);
      return;
    }

    // Process inline formatting
    // Bold: **text**
    const parts = [];
    const boldRegex = /\*\*(.*?)\*\*/g;
    let lastIndex = 0;
    let match;
    
    while ((match = boldRegex.exec(content)) !== null) {
      if (match.index > lastIndex) {
        parts.push(content.substring(lastIndex, match.index));
      }
      parts.push(<strong key={`b-${lineIdx}-${match.index}`} className="font-bold text-slate-800">{match[1]}</strong>);
      lastIndex = boldRegex.lastIndex;
    }
    if (lastIndex < content.length) {
      parts.push(content.substring(lastIndex));
    }
    
    // Code: `text`
    const processedParts = parts.map((part, i) => {
      if (typeof part === 'string') {
        const codeParts = part.split(/`([^`]+)`/);
        return codeParts.map((cp, j) => 
          j % 2 === 1 
            ? <code key={`c-${lineIdx}-${i}-${j}`} className="bg-violet-50 text-violet-700 px-1.5 py-0.5 rounded text-xs font-mono">{cp}</code>
            : cp
        );
      }
      return part;
    });
    
    // Italic: *text*
    const finalParts = processedParts.flat().map((part, i) => {
      if (typeof part === 'string') {
        const italicParts = part.split(/\*([^*]+)\*/);
        return italicParts.map((ip, j) =>
          j % 2 === 1
            ? <em key={`i-${lineIdx}-${i}-${j}`} className="italic text-slate-500">{ip}</em>
            : ip
        );
      }
      return part;
    });

    elements.push(
      <p key={lineIdx} className="text-sm text-slate-600 leading-relaxed">
        {finalParts.flat()}
      </p>
    );
  });
  
  return elements;
};

// Topic icons for visual flair
const topicIcons = {
  'protein': <Dna className="text-teal-500" size={16} />,
  'disease': <HeartPulse className="text-rose-500" size={16} />,
  'drug': <Pill className="text-indigo-500" size={16} />,
  'ai': <BrainCircuit className="text-violet-500" size={16} />,
  'default': <FlaskConical className="text-cyan-500" size={16} />,
};

const getTopicIcon = (text) => {
  const lower = text.toLowerCase();
  if (lower.includes('protein') || lower.includes('gene') || lower.includes('dna')) return topicIcons.protein;
  if (lower.includes('disease') || lower.includes('cancer') || lower.includes('alzheimer')) return topicIcons.disease;
  if (lower.includes('drug') || lower.includes('target') || lower.includes('treatment')) return topicIcons.drug;
  if (lower.includes('transgraph') || lower.includes('model') || lower.includes('shap') || lower.includes('ensemble')) return topicIcons.ai;
  return topicIcons.default;
};

const Assistant = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [predictionContext, setPredictionContext] = useState(null); // injected from Predict page
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load prediction context from localStorage (saved by Predict.jsx on prediction)
  useEffect(() => {
    try {
      const stored = localStorage.getItem('transgraph_last_prediction');
      if (stored) {
        const ctx = JSON.parse(stored);
        setPredictionContext(ctx);
      }
    } catch (_) {}
  }, []);

  // Load greeting on mount
  useEffect(() => {
    const loadGreeting = async () => {
      try {
        const res = await ppiService.getChatGreeting();
        setMessages([{
          type: 'assistant',
          text: res.data.response,
          sources: res.data.sources || [],
          timestamp: new Date()
        }]);
        setSuggestions(res.data.suggestions || []);
      } catch (err) {
        setMessages([{
          type: 'assistant',
          text: "👋 Hello! I'm your Protein Discovery Assistant. Ask me about any protein, disease, or biology concept! For example, try asking 'What is p53?' or 'Tell me about BRCA1'.",
          sources: [],
          timestamp: new Date()
        }]);
        setSuggestions([
          "What is p53?",
          "Tell me about BRCA1",
          "How does TransGraph-PPI work?",
          "What is a protein-protein interaction?"
        ]);
      }
    };
    loadGreeting();
  }, []);

  const handleSend = async (text = null) => {
    const message = text || input.trim();
    if (!message || loading) return;

    // Build context-aware message prefix if a prediction exists
    let contextualMessage = message;
    if (predictionContext) {
      const { p1, p2, prob, esm, gat, conf } = predictionContext;
      contextualMessage = `[Context: The user just ran TransGraph-PPI prediction for ${p1} ↔ ${p2}. Results: interaction_probability=${(prob*100).toFixed(1)}%, ESM=${(esm*100).toFixed(1)}%, GAT=${(gat*100).toFixed(1)}%, confidence=${(conf*100).toFixed(1)}%. Please use this context to answer the following question.]\n\n${message}`;
    }

    // Add user message (show without system prefix)
    const userMsg = { type: 'user', text: message, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setSuggestions([]);
    setLoading(true);

    try {
      const res = await ppiService.sendChatMessage(contextualMessage);
      const assistantMsg = {
        type: 'assistant',
        text: res.data.response,
        sources: res.data.sources || [],
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMsg]);
      setSuggestions(res.data.suggestions || []);
    } catch (err) {
      setMessages(prev => [...prev, {
        type: 'assistant',
        text: "I'm sorry, I couldn't process that request. Please make sure the backend server is running and try again.",
        sources: [],
        timestamp: new Date(),
        isError: true
      }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="max-w-4xl mx-auto h-[calc(100vh-4rem)] flex flex-col">
      {/* Header */}
      <div className="glass-card p-5 mb-4 flex items-center justify-between" style={{ transform: 'none' }}>
        <div className="flex items-center gap-4">
          <div className="relative">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-violet-500 via-purple-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-200">
              <Bot className="text-white" size={24} />
            </div>
            <div className="absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 bg-emerald-400 rounded-full border-2 border-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
              Protein Discovery Assistant
              <Sparkles size={16} className="text-amber-400" />
            </h2>
            <p className="text-xs text-slate-400 font-medium">
              Ask about proteins, diseases, drug targets & biology
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-50 rounded-full border border-emerald-100">
          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
          <span className="text-[10px] font-bold text-emerald-600 uppercase tracking-wider">Online</span>
        </div>
      </div>

      {/* Prediction Context Banner */}
      {predictionContext && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mx-1 mb-3 p-4 bg-teal-50 border border-teal-200 rounded-2xl"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1">
              <p className="text-[10px] font-black text-teal-600 uppercase tracking-widest mb-1.5">
                🔬 Prediction Context Loaded
              </p>
              <p className="text-xs text-teal-800 font-medium">
                <span className="font-mono">{predictionContext.p1}</span>
                <span className="mx-2 text-teal-400">↔</span>
                <span className="font-mono">{predictionContext.p2}</span>
                <span className="ml-3 px-2 py-0.5 bg-teal-100 text-teal-700 rounded-full text-[9px] font-black">
                  {(predictionContext.prob * 100).toFixed(1)}% interaction
                </span>
              </p>
            </div>
            <button
              onClick={() => { setPredictionContext(null); localStorage.removeItem('transgraph_last_prediction'); }}
              className="text-teal-400 hover:text-teal-600 text-[10px] font-bold uppercase tracking-wider"
            >
              Clear
            </button>
          </div>
          <div className="flex flex-wrap gap-2 mt-3">
            {[
              `Why did ${predictionContext.p1} and ${predictionContext.p2} interact?`,
              `What diseases are associated with ${predictionContext.p1}?`,
              `Suggest drug targets for this interaction.`,
              `Explain the biological significance of this pair.`
            ].map((s, i) => (
              <button key={i} onClick={() => handleSend(s)}
                className="px-2.5 py-1 bg-teal-100 hover:bg-teal-200 text-teal-700 rounded-lg text-[9px] font-bold transition-all cursor-pointer">
                {s}
              </button>
            ))}
          </div>
        </motion.div>
      )}

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-1 space-y-4 pb-4 scrollbar-thin" id="chat-messages">
        <AnimatePresence initial={false}>
          {messages.map((msg, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 12, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
              className={`flex gap-3 ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.type === 'assistant' && (
                <div className="flex-shrink-0 mt-1">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shadow-md">
                    <Bot className="text-white" size={16} />
                  </div>
                </div>
              )}

              <div className={`max-w-[80%] ${msg.type === 'user' ? 'order-first' : ''}`}>
                <div className={`rounded-2xl px-5 py-4 ${
                  msg.type === 'user'
                    ? 'bg-gradient-to-br from-teal-500 to-cyan-600 text-white shadow-lg shadow-teal-100'
                    : msg.isError
                      ? 'bg-red-50 border border-red-100'
                      : 'bg-white border border-slate-100 shadow-sm'
                }`}>
                  {msg.type === 'user' ? (
                    <p className="text-sm font-medium leading-relaxed">{msg.text}</p>
                  ) : (
                    <div className="prose-sm">
                      {renderMarkdown(msg.text)}
                    </div>
                  )}
                </div>

                {/* Sources */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="flex items-center gap-2 mt-2 ml-2 flex-wrap">
                    <BookOpen size={10} className="text-slate-300" />
                    {msg.sources.map((src, i) => (
                      <span key={i} className="text-[9px] font-medium text-slate-400 bg-slate-50 px-2 py-0.5 rounded-full border border-slate-100">
                        {src}
                      </span>
                    ))}
                  </div>
                )}

                {/* Timestamp */}
                <p className={`text-[9px] mt-1.5 font-medium ${msg.type === 'user' ? 'text-right text-slate-400' : 'text-slate-300 ml-2'}`}>
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>

              {msg.type === 'user' && (
                <div className="flex-shrink-0 mt-1">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-slate-600 to-slate-800 flex items-center justify-center shadow-md">
                    <User className="text-white" size={16} />
                  </div>
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading bubble */}
        {loading && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex gap-3 justify-start"
          >
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shadow-md flex-shrink-0 mt-1">
              <Bot className="text-white" size={16} />
            </div>
            <div className="bg-white border border-slate-100 shadow-sm rounded-2xl px-5 py-4">
              <div className="flex items-center gap-2">
                <Loader2 size={14} className="animate-spin text-violet-500" />
                <span className="text-xs text-slate-400 font-medium">Searching knowledge base...</span>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggestions */}
      <AnimatePresence>
        {suggestions.length > 0 && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            className="flex flex-wrap gap-2 px-1 pb-3"
          >
            {suggestions.map((s, i) => (
              <motion.button
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.05 }}
                onClick={() => handleSend(s)}
                className="group flex items-center gap-1.5 px-3.5 py-2 bg-white hover:bg-violet-50 border border-slate-200 hover:border-violet-200 rounded-xl text-xs font-medium text-slate-600 hover:text-violet-700 transition-all duration-200 shadow-sm hover:shadow cursor-pointer"
              >
                {getTopicIcon(s)}
                <span>{s}</span>
                <ChevronRight size={12} className="text-slate-300 group-hover:text-violet-400 group-hover:translate-x-0.5 transition-all" />
              </motion.button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Input Area */}
      <div className="glass-card p-3 flex items-center gap-3" style={{ transform: 'none' }}>
        <div className="flex-1 relative">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about any protein, disease, or biology concept..."
            className="w-full pl-4 pr-4 py-3.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-violet-300 focus:border-violet-300 outline-none transition-all text-sm placeholder:text-slate-300"
            disabled={loading}
          />
        </div>
        <button
          onClick={() => handleSend()}
          disabled={loading || !input.trim()}
          className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-violet-500 to-indigo-600 text-white rounded-xl flex items-center justify-center shadow-lg shadow-violet-200 hover:shadow-xl hover:scale-105 active:scale-95 transition-all duration-200 disabled:opacity-40 disabled:shadow-none disabled:hover:scale-100 cursor-pointer"
        >
          {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
        </button>
      </div>
    </div>
  );
};

export default Assistant;

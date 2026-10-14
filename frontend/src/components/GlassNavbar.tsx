import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Menu, WandSparkles, X } from 'lucide-react';
import './GlassNavbar.css';

const NAV_ITEMS = [
  { label: 'How it works', href: '#how-it-works' },
  { label: 'Compression', href: '#compression' },
  { label: 'About', href: '#about' },
];

export default function GlassNavbar() {
  const [active, setActive] = useState<number>(1);
  const [hovered, setHovered] = useState<number | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="glass-nav-wrap">
      <motion.nav
        initial={{ y: -40, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 200, damping: 24 }}
        className="glass-nav"
      >
        {/* Brand — keeps PromptLite content */}
        <a
          className="glass-brand"
          href="#top"
          onClick={() => {
            setActive(1);
            setMenuOpen(false);
          }}
        >
          <span className="brand-mark">
            <WandSparkles size={15} />
          </span>
          <span>PromptLite</span>
        </a>

        <div className="glass-spacer" />

        {/* Desktop links — animated active pill like glass-navbar.json */}
        <div className="glass-links">
          {NAV_ITEMS.map((item, i) => (
            <motion.a
              key={item.label}
              href={item.href}
              onClick={() => setActive(i)}
              onHoverStart={() => setHovered(i)}
              onHoverEnd={() => setHovered(null)}
              className={`glass-link${active === i || hovered === i ? ' is-on' : ''}`}
              whileTap={{ scale: 0.97 }}
            >
              {active === i && (
                <motion.span
                  layoutId="glass-nav-active"
                  className="glass-link-pill"
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
              <span className="glass-link-label">{item.label}</span>
            </motion.a>
          ))}

          <span className="glass-status">
            <i />
            100% Local · No API Key
          </span>

          <motion.a
            href="#compression"
            onClick={() => setActive(1)}
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            className="glass-cta"
            transition={{ duration: 0.25 }}
          >
            Compress Now
          </motion.a>
        </div>

        {/* Mobile toggle */}
        <motion.button
          className="glass-menu-btn"
          onClick={() => setMenuOpen((v) => !v)}
          whileTap={{ scale: 0.9 }}
          aria-label="Toggle menu"
        >
          <AnimatePresence mode="wait" initial={false}>
            {menuOpen ? (
              <motion.span
                key="x"
                initial={{ rotate: -90, opacity: 0 }}
                animate={{ rotate: 0, opacity: 1 }}
                exit={{ rotate: 90, opacity: 0 }}
                transition={{ duration: 0.15 }}
              >
                <X size={18} />
              </motion.span>
            ) : (
              <motion.span
                key="menu"
                initial={{ rotate: 90, opacity: 0 }}
                animate={{ rotate: 0, opacity: 1 }}
                exit={{ rotate: -90, opacity: 0 }}
                transition={{ duration: 0.15 }}
              >
                <Menu size={18} />
              </motion.span>
            )}
          </AnimatePresence>
        </motion.button>
      </motion.nav>

      {/* Mobile dropdown */}
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.97 }}
            transition={{ duration: 0.22 }}
            className="glass-mobile"
          >
            {NAV_ITEMS.map((item, i) => (
              <a
                key={item.label}
                href={item.href}
                onClick={() => {
                  setActive(i);
                  setMenuOpen(false);
                }}
                className={`glass-mobile-link${active === i ? ' is-on' : ''}`}
              >
                {item.label}
              </a>
            ))}
            <a
              href="#compression"
              onClick={() => {
                setActive(1);
                setMenuOpen(false);
              }}
              className="glass-cta glass-mobile-cta"
            >
              Compress Now
            </a>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

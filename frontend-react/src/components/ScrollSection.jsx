import { motion } from 'framer-motion';
import useScrollAnimation from '../hooks/useScrollAnimation';

const variants = {
  hidden: { opacity: 0, y: 60 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.7, ease: 'easeOut' } },
};

export default function ScrollSection({ children, className = '', delay = 0 }) {
  const [ref, isVisible] = useScrollAnimation(0.15);

  return (
    <motion.section
      ref={ref}
      className={className}
      initial="hidden"
      animate={isVisible ? 'visible' : 'hidden'}
      variants={{
        ...variants,
        visible: {
          ...variants.visible,
          transition: { ...variants.visible.transition, delay },
        },
      }}
    >
      {children}
    </motion.section>
  );
}

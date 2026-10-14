export type Mode = 'safe' | 'balanced' | 'aggressive';
export type CompressionResult = { optimized_text: string; original_tokens: number; optimized_tokens: number; token_reduction_percentage: number; compression_ratio: number; semantic_similarity: number; detected_task: string; protected_items: string[]; removed_items_summary: string[]; warnings: string[]; mode_used: Mode };

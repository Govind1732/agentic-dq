import Joi from 'joi';
import DOMPurify from 'dompurify';

// Client-side validation schemas matching backend
export const rcaDataSchema = Joi.object({
  failed_table: Joi.string().min(1).max(255).required(),
  failed_column: Joi.string().min(1).max(255).required(),
  db_type: Joi.string().valid('GCP', 'Teradata').required(),
  validation_query: Joi.string().min(1).max(10000).required(),
  sd_threshold: Joi.alternatives().try(Joi.string().allow(''), Joi.number()),
  expected_std_dev: Joi.alternatives().try(Joi.string().allow(''), Joi.number()),
  expected_value: Joi.alternatives().try(Joi.string().allow(''), Joi.number()),
  actual_value: Joi.alternatives().try(Joi.string().allow(''), Joi.number()),
  execution_date: Joi.alternatives().try(Joi.string().allow(''), Joi.date())
}).unknown(true);

export const messageSchema = Joi.object({
  message: Joi.alternatives().try(
    Joi.string().min(1).max(50000),
    rcaDataSchema,
    Joi.object().unknown(true)
  ).required(),
  timestamp: Joi.date().iso().optional()
}).unknown(true);

// Input sanitization
export const sanitizeInput = (input) => {
  if (typeof input === 'string') {
    return DOMPurify.sanitize(input);
  }
  
  if (typeof input === 'object' && input !== null) {
    const sanitized = {};
    for (const [key, value] of Object.entries(input)) {
      if (typeof value === 'string') {
        sanitized[key] = DOMPurify.sanitize(value);
      } else if (typeof value === 'object' && value !== null) {
        sanitized[key] = sanitizeInput(value);
      } else {
        sanitized[key] = value;
      }
    }
    return sanitized;
  }
  
  return input;
};

// Validation wrapper
export const validateInput = (data, schema) => {
  const { error, value } = schema.validate(data, {
    stripUnknown: true,
    abortEarly: false
  });
  
  if (error) {
    throw new Error(`Validation error: ${error.details.map(d => d.message).join(', ')}`);
  }
  
  return value;
};

// Combined validation and sanitization
export const validateAndSanitize = (data, schema) => {
  const sanitized = sanitizeInput(data);
  return validateInput(sanitized, schema);
};

// Rate limiting for client-side
export class ClientRateLimit {
  constructor(maxRequests = 30, windowMs = 60000) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.requests = [];
  }

  checkLimit() {
    const now = Date.now();
    const windowStart = now - this.windowMs;
    
    // Remove old requests
    this.requests = this.requests.filter(timestamp => timestamp > windowStart);
    
    if (this.requests.length >= this.maxRequests) {
      return false;
    }
    
    this.requests.push(now);
    return true;
  }

  getTimeUntilReset() {
    if (this.requests.length === 0) return 0;
    
    const oldestRequest = Math.min(...this.requests);
    const resetTime = oldestRequest + this.windowMs;
    return Math.max(0, resetTime - Date.now());
  }
}

// Message size validation
export const validateMessageSize = (message, maxSize = 50000) => {
  const messageStr = typeof message === 'string' ? message : JSON.stringify(message);
  if (messageStr.length > maxSize) {
    throw new Error(`Message too large. Maximum size is ${maxSize} characters.`);
  }
  return true;
};

// JSON validation helper
export const validateJSON = (jsonString) => {
  try {
    const parsed = JSON.parse(jsonString);
    return { valid: true, data: parsed };
  } catch (error) {
    return { valid: false, error: error.message };
  }
};

// SQL injection detection (refined to be less aggressive)
export const detectSQLInjection = (input) => {
  const sqlPatterns = [
    // More specific SQL injection patterns
    /(\bSELECT\b.*\bFROM\b)|(\bINSERT\b.*\bINTO\b)|(\bUPDATE\b.*\bSET\b)|(\bDELETE\b.*\bFROM\b)/i,
    /(\bDROP\b\s+(TABLE|DATABASE|SCHEMA))|(\bCREATE\b\s+(TABLE|DATABASE))/i,
    /(\bUNION\b\s+(SELECT|ALL))|(\bEXEC\b\s*\()|(\bSCRIPT\b)/i,
    // Dangerous comment patterns
    /(--|\/\*.*\*\/)/,
    // SQL injection specific patterns
    /('\s*(OR|AND)\s*'.*')|('\s*(OR|AND)\s*\d+\s*=\s*\d+)/i
  ];
  
  return sqlPatterns.some(pattern => pattern.test(input));
};

// XSS detection (refined)
export const detectXSS = (input) => {
  const xssPatterns = [
    /<script[\s\S]*?>[\s\S]*?<\/script>/gi,
    /<iframe[\s\S]*?>[\s\S]*?<\/iframe>/gi,
    /javascript\s*:/gi,
    /on(click|load|error|mouseover|focus|blur)\s*=/gi,
    /<img[\s\S]*onerror[\s\S]*>/gi,
    /<object[\s\S]*?>[\s\S]*?<\/object>/gi,
    /<embed[\s\S]*?>[\s\S]*?<\/embed>/gi
  ];
  
  return xssPatterns.some(pattern => pattern.test(input));
};

export default {
  rcaDataSchema,
  messageSchema,
  sanitizeInput,
  validateInput,
  validateAndSanitize,
  ClientRateLimit,
  validateMessageSize,
  validateJSON,
  detectSQLInjection,
  detectXSS
};

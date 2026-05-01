#ifndef FROTZ_LLM_H_
#define FROTZ_LLM_H_

#include <stddef.h>

typedef enum llm_provider {
    LLM_PROVIDER_NONE = 0,
    LLM_PROVIDER_OPENAI,
    LLM_PROVIDER_OPENROUTER,
    LLM_PROVIDER_LMSTUDIO,
    LLM_PROVIDER_ANTHROPIC,
    LLM_PROVIDER_GITHUB_MODELS
} llm_provider_t;

int llm_is_configured(void);
int llm_load_config(const char *path_hint, char *error, size_t error_size);
int llm_mode_is_test(void);
int llm_mode_is_interactive(void);
int llm_agent_start(const char *goal, char *error, size_t error_size);
void llm_agent_stop(void);
int llm_agent_is_active(void);
int llm_agent_next_command(char *command,
                           size_t command_size,
                           char *status,
                           size_t status_size,
                           char *error,
                           size_t error_size);
void llm_capture_output_word(const unsigned char *s);
void llm_capture_output_char(unsigned char c);
void llm_capture_output_new_line(void);
void llm_capture_user_input(const unsigned char *s, unsigned char terminator);
int llm_query_with_context(const char *query,
                           char *response,
                           size_t response_size,
                           char *error,
                           size_t error_size);
long llm_last_latency_ms(void);
int llm_last_attempts(void);
int llm_run_prompt(const char *provider_hint,
                   const char *model_hint,
                   const char *base_url_hint,
                   const char *api_key_hint,
                   const char *prompt,
                   char *response,
                   size_t response_size,
                   char *error,
                   size_t error_size);

#endif

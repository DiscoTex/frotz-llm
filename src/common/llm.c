#include "frotz.h"
#include "llm.h"
#include "jsmn.h"

#include <ctype.h>
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#define LLM_RESPONSE_BUF_SIZE 1048576
#define LLM_CMD_BUF_SIZE 8192
#define LLM_PATH_BUF_SIZE 512
#define LLM_MAX_TOKENS 4096
#define LLM_CAPTURE_LINES 256
#define LLM_CAPTURE_CMDS 64
#define LLM_CAPTURE_LINE_LEN 256

static char llm_output_lines[LLM_CAPTURE_LINES][LLM_CAPTURE_LINE_LEN];
static int llm_output_next = 0;
static int llm_output_count = 0;
static char llm_cmd_lines[LLM_CAPTURE_CMDS][LLM_CAPTURE_LINE_LEN];
static int llm_cmd_next = 0;
static int llm_cmd_count = 0;
static char llm_current_line[LLM_CAPTURE_LINE_LEN];
static int llm_current_line_len = 0;
static int llm_agent_active = 0;
static int llm_agent_step = 0;
static int llm_agent_max_steps = 8;
static char llm_agent_goal[1024];
static long llm_last_latency_ms_value = 0;
static int llm_last_attempts_value = 0;
static volatile sig_atomic_t llm_curl_cancelled = 0;

static long long llm_now_ms(void)
{
    struct timeval tv;
    (void)gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000LL + (long long)(tv.tv_usec / 1000);
}

static int llm_verbose_level(void)
{
    const char *v = getenv("FROTZ_LLM_VERBOSE");

    if (v == NULL || *v == '\0')
        return 0;
    if (v[0] == '0')
        return 0;
    if (v[0] == '2')
        return 2;
    if (v[0] == '3')
        return 3;
    return 1;
}

static void llm_log_block(int level, const char *title, const char *text)
{
    const char *to_stderr;
    FILE *out;

    if (llm_verbose_level() < level)
        return;

    to_stderr = getenv("FROTZ_LLM_VERBOSE_STDERR");
    if (to_stderr != NULL && *to_stderr != '\0' && to_stderr[0] != '0') {
        out = stderr;
    } else {
        out = fopen("/tmp/frotz-llm.log", "a");
        if (out == NULL)
            out = stderr;
    }

    fprintf(out, "[LLM] BEGIN %s\n", title != NULL ? title : "BLOCK");
    if (text != NULL)
        fputs(text, out);
    fputs("\n", out);
    fprintf(out, "[LLM] END %s\n", title != NULL ? title : "BLOCK");
    fflush(out);

    if (out != stderr)
        fclose(out);
}

static void llm_logf(int level, const char *fmt, ...)
{
    va_list ap;
    const char *to_stderr;
    FILE *out;

    if (llm_verbose_level() < level)
        return;

    to_stderr = getenv("FROTZ_LLM_VERBOSE_STDERR");
    if (to_stderr != NULL && *to_stderr != '\0' && to_stderr[0] != '0') {
        out = stderr;
    } else {
        out = fopen("/tmp/frotz-llm.log", "a");
        if (out == NULL)
            out = stderr;
    }

    fputs("[LLM] ", out);
    va_start(ap, fmt);
    vfprintf(out, fmt, ap);
    va_end(ap);
    fputs("\n", out);
    fflush(out);

    if (out != stderr)
        fclose(out);
}

static int str_ieq(const char *a, const char *b)
{
    if (a == NULL || b == NULL)
        return 0;

    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
            return 0;
        a++;
        b++;
    }

    return *a == '\0' && *b == '\0';
}

static llm_provider_t llm_provider_from_string(const char *provider)
{
    if (provider == NULL || *provider == '\0')
        return LLM_PROVIDER_NONE;
    if (str_ieq(provider, "openai"))
        return LLM_PROVIDER_OPENAI;
    if (str_ieq(provider, "openrouter"))
        return LLM_PROVIDER_OPENROUTER;
    if (str_ieq(provider, "lmstudio") || str_ieq(provider, "lm-studio"))
        return LLM_PROVIDER_LMSTUDIO;
    if (str_ieq(provider, "anthropic") || str_ieq(provider, "claude"))
        return LLM_PROVIDER_ANTHROPIC;
    if (str_ieq(provider, "github-models") || str_ieq(provider, "github_models"))
        return LLM_PROVIDER_GITHUB_MODELS;

    return LLM_PROVIDER_NONE;
}

static const char *llm_default_model(llm_provider_t provider)
{
    switch (provider) {
    case LLM_PROVIDER_OPENAI:
        return "gpt-4o-mini";
    case LLM_PROVIDER_OPENROUTER:
        return "openai/gpt-4o-mini";
    case LLM_PROVIDER_LMSTUDIO:
        return "local-model";
    case LLM_PROVIDER_ANTHROPIC:
        return "claude-3-5-haiku-latest";
    case LLM_PROVIDER_GITHUB_MODELS:
        return "openai/gpt-4.1";
    default:
        return "";
    }
}

static const char *llm_default_endpoint(llm_provider_t provider)
{
    switch (provider) {
    case LLM_PROVIDER_OPENAI:
        return "https://api.openai.com/v1/chat/completions";
    case LLM_PROVIDER_OPENROUTER:
        return "https://openrouter.ai/api/v1/chat/completions";
    case LLM_PROVIDER_LMSTUDIO:
        return "http://127.0.0.1:1234/v1/chat/completions";
    case LLM_PROVIDER_ANTHROPIC:
        return "https://api.anthropic.com/v1/messages";
    case LLM_PROVIDER_GITHUB_MODELS:
        return "https://models.github.ai/inference/chat/completions";
    default:
        return "";
    }
}

static const char *first_nonempty(const char *a, const char *b, const char *c)
{
    if (a != NULL && *a != '\0')
        return a;
    if (b != NULL && *b != '\0')
        return b;
    if (c != NULL && *c != '\0')
        return c;
    return NULL;
}

static size_t json_escape(const char *in, char *out, size_t out_size)
{
    size_t i = 0;
    size_t o = 0;

    if (out_size == 0)
        return 0;

    while (in[i] != '\0' && o + 2 < out_size) {
        unsigned char c = (unsigned char)in[i++];

        switch (c) {
        case '\\':
            out[o++] = '\\';
            out[o++] = '\\';
            break;
        case '"':
            out[o++] = '\\';
            out[o++] = '"';
            break;
        case '\n':
            out[o++] = '\\';
            out[o++] = 'n';
            break;
        case '\r':
            out[o++] = '\\';
            out[o++] = 'r';
            break;
        case '\t':
            out[o++] = '\\';
            out[o++] = 't';
            break;
        default:
            if (c < 0x20) {
                if (o + 6 >= out_size) {
                    out[o] = '\0';
                    return o;
                }
                (void)snprintf(out + o, out_size - o, "\\u%04x", c);
                o += 6;
            } else {
                out[o++] = (char)c;
            }
            break;
        }
    }

    out[o] = '\0';
    return o;
}

static void shell_append_quoted(char *dst, size_t dst_size, const char *value)
{
    size_t dlen = strlen(dst);
    size_t i;

    if (dlen + 2 >= dst_size)
        return;

    dst[dlen++] = '\'';
    dst[dlen] = '\0';

    for (i = 0; value[i] != '\0'; i++) {
        if (value[i] == '\'') {
            if (dlen + 4 >= dst_size)
                break;
            memcpy(dst + dlen, "'\\''", 4);
            dlen += 4;
        } else {
            if (dlen + 1 >= dst_size)
                break;
            dst[dlen++] = value[i];
        }
    }

    if (dlen + 1 < dst_size) {
        dst[dlen++] = '\'';
        dst[dlen] = '\0';
    }
}

static void shell_append_header(char *dst, size_t dst_size, const char *name, const char *value)
{
    char header[1024];

    (void)snprintf(header, sizeof(header), "%s: %s", name, value);
    strncat(dst, " -H ", dst_size - strlen(dst) - 1);
    shell_append_quoted(dst, dst_size, header);
}

static int write_temp_payload(const char *payload, char *path, size_t path_size, char *error, size_t error_size)
{
    int fd;
    FILE *fp;
    char template_path[] = "/tmp/frotz_llm_XXXXXX";

    if (path_size < sizeof(template_path)) {
        (void)snprintf(error, error_size, "temporary path buffer too small");
        return 0;
    }

    fd = mkstemp(template_path);
    if (fd < 0) {
        (void)snprintf(error, error_size, "unable to create temporary payload file: %s", strerror(errno));
        return 0;
    }

    fp = fdopen(fd, "w");
    if (fp == NULL) {
        close(fd);
        (void)snprintf(error, error_size, "unable to write payload file: %s", strerror(errno));
        return 0;
    }

    if (fputs(payload, fp) == EOF) {
        fclose(fp);
        unlink(template_path);
        (void)snprintf(error, error_size, "unable to write payload data");
        return 0;
    }

    if (fclose(fp) != 0) {
        unlink(template_path);
        (void)snprintf(error, error_size, "unable to finalize payload file");
        return 0;
    }

    strncpy(path, template_path, path_size - 1);
    path[path_size - 1] = '\0';
    return 1;
}

static void llm_cancel_handler(int sig)
{
    (void)sig;
    llm_curl_cancelled = 1;
}

static int run_curl_command(const char *cmd, char *body, size_t body_size, char *error, size_t error_size)
{
    int pipefd[2];
    pid_t child;
    struct sigaction sa_new, sa_old;
    size_t used = 0;
    int child_status = 0;
    int cancelled = 0;

    if (body_size == 0) {
        (void)snprintf(error, error_size, "body buffer too small");
        return 0;
    }
    body[0] = '\0';

    if (pipe(pipefd) != 0) {
        (void)snprintf(error, error_size, "pipe failed: %s", strerror(errno));
        return 0;
    }

    /* Intercept SIGINT while waiting so Ctrl+C cancels the request. */
    llm_curl_cancelled = 0;
    sa_new.sa_handler = llm_cancel_handler;
    sigemptyset(&sa_new.sa_mask);
    sa_new.sa_flags = 0;
    sigaction(SIGINT, &sa_new, &sa_old);

    child = fork();
    if (child < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        sigaction(SIGINT, &sa_old, NULL);
        (void)snprintf(error, error_size, "fork failed: %s", strerror(errno));
        return 0;
    }

    if (child == 0) {
        /* Child: redirect stdout+stderr to pipe, exec shell. */
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
        _exit(127);
    }

    /* Parent: read from pipe with select so SIGINT can interrupt. */
    close(pipefd[1]);

    while (!llm_curl_cancelled) {
        fd_set rfds;
        struct timeval tv;
        int rc;

        FD_ZERO(&rfds);
        FD_SET(pipefd[0], &rfds);
        tv.tv_sec = 0;
        tv.tv_usec = 100000; /* 100 ms poll interval */

        rc = select(pipefd[0] + 1, &rfds, NULL, NULL, &tv);
        if (rc < 0) {
            if (errno == EINTR)
                break; /* signal received */
            break;
        }
        if (rc == 0)
            continue; /* timeout, loop to check cancel flag */

        if (FD_ISSET(pipefd[0], &rfds)) {
            char tmp[4096];
            ssize_t n = read(pipefd[0], tmp, sizeof(tmp));
            if (n <= 0)
                break; /* EOF or error */
            if (used + (size_t)n + 1 < body_size) {
                memcpy(body + used, tmp, (size_t)n);
                used += (size_t)n;
            }
        }
    }
    body[used] = '\0';
    close(pipefd[0]);

    if (llm_curl_cancelled) {
        cancelled = 1;
        kill(child, SIGTERM);
    }

    waitpid(child, &child_status, 0);

    /* Restore original SIGINT handler. */
    sigaction(SIGINT, &sa_old, NULL);

    if (cancelled) {
        (void)snprintf(error, error_size, "LLM request cancelled");
        return 0;
    }

    if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
        (void)snprintf(error, error_size, "curl request failed: %s", body);
        return 0;
    }

    return 1;
}

static int token_equals(const char *json, const jsmntok_t *tok, const char *s)
{
    size_t len = strlen(s);

    if (tok->type != JSMN_STRING)
        return 0;

    if ((size_t)(tok->end - tok->start) != len)
        return 0;

    return strncmp(json + tok->start, s, len) == 0;
}

static int skip_token(jsmntok_t *tokens, int idx)
{
    int i;
    int j = idx + 1;

    if (tokens[idx].type == JSMN_ARRAY) {
        for (i = 0; i < tokens[idx].size; i++)
            j = skip_token(tokens, j);
    } else if (tokens[idx].type == JSMN_OBJECT) {
        for (i = 0; i < tokens[idx].size; i++) {
            j = skip_token(tokens, j);
            j = skip_token(tokens, j);
        }
    }

    return j;
}

static int object_get(const char *json, jsmntok_t *tokens, int obj_idx, const char *key)
{
    int i;
    int j;

    if (tokens[obj_idx].type != JSMN_OBJECT)
        return -1;

    j = obj_idx + 1;
    for (i = 0; i < tokens[obj_idx].size; i++) {
        int key_idx = j;
        int val_idx = key_idx + 1;

        if (token_equals(json, &tokens[key_idx], key))
            return val_idx;

        j = skip_token(tokens, val_idx);
    }

    return -1;
}

static int array_get(jsmntok_t *tokens, int arr_idx, int elem)
{
    int i;
    int j;

    if (tokens[arr_idx].type != JSMN_ARRAY || elem < 0 || elem >= tokens[arr_idx].size)
        return -1;

    j = arr_idx + 1;
    for (i = 0; i < elem; i++)
        j = skip_token(tokens, j);

    return j;
}

static int json_unescape_string(const char *in, size_t len, char *out, size_t out_size)
{
    unsigned long codepoint;
    size_t i = 0;
    size_t o = 0;

    if (out_size == 0)
        return 0;

    while (i < len && o + 1 < out_size) {
        if (in[i] == '\\' && i + 1 < len) {
            i++;
            switch (in[i]) {
            case 'n': out[o++] = '\n'; break;
            case 'r': out[o++] = '\r'; break;
            case 't': out[o++] = '\t'; break;
            case '"':
            case '\\':
            case '/': out[o++] = in[i]; break;
            case 'u': {
                unsigned long hi = 0;
                unsigned long lo = 0;
                size_t j;

                /* Parse exactly four hex digits for \uXXXX. */
                if (i + 4 >= len) {
                    out[o++] = ' ';
                    break;
                }
                for (j = 0; j < 4; j++) {
                    unsigned char c = (unsigned char)in[i + 1 + j];
                    if (!isxdigit(c)) {
                        hi = 0;
                        break;
                    }
                    hi <<= 4;
                    if (c >= '0' && c <= '9')      hi += (unsigned long)(c - '0');
                    else if (c >= 'a' && c <= 'f') hi += (unsigned long)(c - 'a' + 10);
                    else                            hi += (unsigned long)(c - 'A' + 10);
                }
                if (j != 4) {
                    out[o++] = ' ';
                    break;
                }
                i += 4;

                codepoint = hi;

                /* Combine UTF-16 surrogate pairs when present. */
                if (hi >= 0xD800 && hi <= 0xDBFF) {
                    if (i + 6 < len && in[i + 1] == '\\' && in[i + 2] == 'u') {
                        for (j = 0; j < 4; j++) {
                            unsigned char c = (unsigned char)in[i + 3 + j];
                            if (!isxdigit(c)) {
                                lo = 0;
                                break;
                            }
                            lo <<= 4;
                            if (c >= '0' && c <= '9')      lo += (unsigned long)(c - '0');
                            else if (c >= 'a' && c <= 'f') lo += (unsigned long)(c - 'a' + 10);
                            else                            lo += (unsigned long)(c - 'A' + 10);
                        }
                        if (j == 4 && lo >= 0xDC00 && lo <= 0xDFFF) {
                            codepoint = 0x10000 + (((hi - 0xD800) << 10) | (lo - 0xDC00));
                            i += 6; /* consumed \\uXXXX of low surrogate */
                        } else {
                            codepoint = ' ';
                        }
                    } else {
                        codepoint = ' ';
                    }
                } else if (hi >= 0xDC00 && hi <= 0xDFFF) {
                    codepoint = ' ';
                }

                /* Some providers leak CP1252 punctuation as \u0092, \u0097, etc.
                 * Remap the common C1 range to intended Unicode punctuation. */
                if (codepoint >= 0x80 && codepoint <= 0x9F) {
                    switch (codepoint) {
                    case 0x82: codepoint = 0x201A; break;
                    case 0x83: codepoint = 0x0192; break;
                    case 0x84: codepoint = 0x201E; break;
                    case 0x85: codepoint = 0x2026; break;
                    case 0x86: codepoint = 0x2020; break;
                    case 0x87: codepoint = 0x2021; break;
                    case 0x88: codepoint = 0x02C6; break;
                    case 0x89: codepoint = 0x2030; break;
                    case 0x8A: codepoint = 0x0160; break;
                    case 0x8B: codepoint = 0x2039; break;
                    case 0x8C: codepoint = 0x0152; break;
                    case 0x8E: codepoint = 0x017D; break;
                    case 0x91: codepoint = 0x2018; break;
                    case 0x92: codepoint = 0x2019; break;
                    case 0x93: codepoint = 0x201C; break;
                    case 0x94: codepoint = 0x201D; break;
                    case 0x95: codepoint = 0x2022; break;
                    case 0x96: codepoint = 0x2013; break;
                    case 0x97: codepoint = 0x2014; break;
                    case 0x98: codepoint = 0x02DC; break;
                    case 0x99: codepoint = 0x2122; break;
                    case 0x9A: codepoint = 0x0161; break;
                    case 0x9B: codepoint = 0x203A; break;
                    case 0x9C: codepoint = 0x0153; break;
                    case 0x9E: codepoint = 0x017E; break;
                    case 0x9F: codepoint = 0x0178; break;
                    default:   codepoint = ' '; break;
                    }
                }

                /* Emit UTF-8 bytes, forcing invalid/control code points to space. */
                if (codepoint < 0x20) {
                    out[o++] = ' ';
                } else if (codepoint < 0x80) {
                    out[o++] = (char)codepoint;
                } else if (codepoint < 0x800 && o + 2 < out_size) {
                    out[o++] = (char)(0xC0 | (codepoint >> 6));
                    out[o++] = (char)(0x80 | (codepoint & 0x3F));
                } else if (codepoint < 0x10000 && o + 3 < out_size) {
                    out[o++] = (char)(0xE0 | (codepoint >> 12));
                    out[o++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
                    out[o++] = (char)(0x80 | (codepoint & 0x3F));
                } else if (codepoint <= 0x10FFFF && o + 4 < out_size) {
                    out[o++] = (char)(0xF0 | (codepoint >> 18));
                    out[o++] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
                    out[o++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
                    out[o++] = (char)(0x80 | (codepoint & 0x3F));
                } else {
                    out[o++] = ' ';
                }
                break;
            }
            default:
                out[o++] = in[i];
                break;
            }
            i++;
        } else {
            out[o++] = in[i++];
        }
    }

    out[o] = '\0';
    return o > 0;
}

static int token_to_string(const char *json, jsmntok_t *tok, char *out, size_t out_size)
{
    if (tok->type != JSMN_STRING)
        return 0;

    return json_unescape_string(json + tok->start,
                                (size_t)(tok->end - tok->start),
                                out,
                                out_size);
}

static int extract_json_string_field_loose(const char *json,
                                           const char *field,
                                           char *out,
                                           size_t out_size)
{
    char needle[64];
    const char *p;

    if (json == NULL || field == NULL || out == NULL || out_size == 0)
        return 0;

    (void)snprintf(needle, sizeof(needle), "\"%s\"", field);
    p = json;

    while ((p = strstr(p, needle)) != NULL) {
        const char *colon = p + strlen(needle);
        const char *q;
        const char *start;
        int escaped = 0;
        char raw[8192];
        size_t raw_len;

        while (*colon != '\0' && isspace((unsigned char)*colon))
            colon++;
        if (*colon != ':') {
            p += strlen(needle);
            continue;
        }
        colon++;

        while (*colon != '\0' && isspace((unsigned char)*colon))
            colon++;
        if (*colon != '"') {
            p += strlen(needle);
            continue;
        }

        start = ++colon;
        q = start;
        while (*q != '\0') {
            if (escaped) {
                escaped = 0;
            } else if (*q == '\\') {
                escaped = 1;
            } else if (*q == '"') {
                break;
            }
            q++;
        }

        if (*q != '"')
            return 0;

        raw_len = (size_t)(q - start);
        if (raw_len >= sizeof(raw))
            raw_len = sizeof(raw) - 1;
        memcpy(raw, start, raw_len);
        raw[raw_len] = '\0';

        return json_unescape_string(raw, raw_len, out, out_size);
    }

    return 0;
}

static int extract_json_response_text(const char *json, char *out, size_t out_size)
{
    jsmn_parser parser;
    jsmntok_t tokens[LLM_MAX_TOKENS];
    int tok_count;
    int choices_idx;
    int choice0_idx;
    int message_idx;
    int content_idx;

    jsmn_init(&parser);
    tok_count = jsmn_parse(&parser, json, strlen(json), tokens, LLM_MAX_TOKENS);
    if (tok_count < 1 || tokens[0].type != JSMN_OBJECT)
        goto loose_fallback;

    choices_idx = object_get(json, tokens, 0, "choices");
    if (choices_idx >= 0) {
        choice0_idx = array_get(tokens, choices_idx, 0);
        if (choice0_idx >= 0) {
            message_idx = object_get(json, tokens, choice0_idx, "message");
            if (message_idx >= 0) {
                content_idx = object_get(json, tokens, message_idx, "content");
                if (content_idx >= 0) {
                    if (tokens[content_idx].type == JSMN_STRING)
                        return token_to_string(json, &tokens[content_idx], out, out_size);
                    if (tokens[content_idx].type == JSMN_ARRAY) {
                        int part0 = array_get(tokens, content_idx, 0);
                        int text_idx;
                        if (part0 >= 0 && tokens[part0].type == JSMN_OBJECT) {
                            text_idx = object_get(json, tokens, part0, "text");
                            if (text_idx >= 0)
                                return token_to_string(json, &tokens[text_idx], out, out_size);
                        }
                    }
                }
            }
        }
    }

    content_idx = object_get(json, tokens, 0, "content");
    if (content_idx >= 0) {
        if (tokens[content_idx].type == JSMN_STRING)
            return token_to_string(json, &tokens[content_idx], out, out_size);
        if (tokens[content_idx].type == JSMN_ARRAY) {
            int part0 = array_get(tokens, content_idx, 0);
            int text_idx;
            if (part0 >= 0 && tokens[part0].type == JSMN_OBJECT) {
                text_idx = object_get(json, tokens, part0, "text");
                if (text_idx >= 0)
                    return token_to_string(json, &tokens[text_idx], out, out_size);
            }
        }
    }

    content_idx = object_get(json, tokens, 0, "text");
    if (content_idx >= 0)
        return token_to_string(json, &tokens[content_idx], out, out_size);

loose_fallback:
    if (extract_json_string_field_loose(json, "content", out, out_size))
        return 1;
    if (extract_json_string_field_loose(json, "text", out, out_size))
        return 1;

    return 0;
}

static int token_to_bool(const char *json, jsmntok_t *tok, int *value)
{
    size_t len;

    if (tok->type != JSMN_PRIMITIVE || value == NULL)
        return 0;

    len = (size_t)(tok->end - tok->start);
    if (len == 4 && strncmp(json + tok->start, "true", 4) == 0) {
        *value = 1;
        return 1;
    }
    if (len == 5 && strncmp(json + tok->start, "false", 5) == 0) {
        *value = 0;
        return 1;
    }

    return 0;
}

static int extract_json_bool_field_loose(const char *json, const char *field, int *value)
{
    char needle[64];
    const char *p;

    if (json == NULL || field == NULL || value == NULL)
        return 0;

    (void)snprintf(needle, sizeof(needle), "\"%s\"", field);
    p = strstr(json, needle);
    if (p == NULL)
        return 0;

    p += strlen(needle);
    while (*p != '\0' && isspace((unsigned char)*p))
        p++;
    if (*p != ':')
        return 0;
    p++;
    while (*p != '\0' && isspace((unsigned char)*p))
        p++;

    if (strncmp(p, "true", 4) == 0) {
        *value = 1;
        return 1;
    }
    if (strncmp(p, "false", 5) == 0) {
        *value = 0;
        return 1;
    }

    return 0;
}

static int extract_first_json_object(const char *text,
                                     char *json_out,
                                     size_t json_out_size)
{
    const char *start;
    const char *p;
    int depth = 0;
    int in_string = 0;
    int escaped = 0;
    size_t len;

    if (text == NULL || json_out == NULL || json_out_size == 0)
        return 0;

    start = strchr(text, '{');
    if (start == NULL)
        return 0;

    for (p = start; *p != '\0'; p++) {
        char c = *p;

        if (in_string) {
            if (escaped) {
                escaped = 0;
            } else if (c == '\\') {
                escaped = 1;
            } else if (c == '"') {
                in_string = 0;
            }
            continue;
        }

        if (c == '"') {
            in_string = 1;
            continue;
        }
        if (c == '{') {
            depth++;
            continue;
        }
        if (c == '}') {
            depth--;
            if (depth == 0) {
                len = (size_t)(p - start + 1);
                if (len >= json_out_size)
                    len = json_out_size - 1;
                memcpy(json_out, start, len);
                json_out[len] = '\0';
                return 1;
            }
        }
    }

    return 0;
}

static int parse_agent_json_reply(const char *reply,
                                  int *done,
                                  char *action,
                                  size_t action_size,
                                  char *summary,
                                  size_t summary_size)
{
    jsmn_parser parser;
    jsmntok_t tokens[256];
    char extracted_json[8192];
    const char *json = reply;
    int tok_count;
    int done_idx;
    int action_idx;
    int summary_idx;

    if (done == NULL || action == NULL || summary == NULL)
        return 0;

    *done = 0;
    action[0] = '\0';
    summary[0] = '\0';

    if (extract_first_json_object(reply, extracted_json, sizeof(extracted_json)))
        json = extracted_json;

    jsmn_init(&parser);
    tok_count = jsmn_parse(&parser, json, strlen(json), tokens, 256);
    if (tok_count < 1 || tokens[0].type != JSMN_OBJECT)
        goto loose_fallback;

    done_idx = object_get(json, tokens, 0, "done");
    if (done_idx >= 0)
        (void)token_to_bool(json, &tokens[done_idx], done);

    action_idx = object_get(json, tokens, 0, "action");
    if (action_idx >= 0 && tokens[action_idx].type == JSMN_STRING)
        (void)token_to_string(json, &tokens[action_idx], action, action_size);

    summary_idx = object_get(json, tokens, 0, "summary");
    if (summary_idx >= 0 && tokens[summary_idx].type == JSMN_STRING)
        (void)token_to_string(json, &tokens[summary_idx], summary, summary_size);

    if (action[0] != '\0' || summary[0] != '\0' || *done)
        return 1;

loose_fallback:
    (void)extract_json_bool_field_loose(reply, "done", done);
    (void)extract_json_string_field_loose(reply, "action", action, action_size);
    (void)extract_json_string_field_loose(reply, "summary", summary, summary_size);

    if (action[0] != '\0' || summary[0] != '\0' || *done)
        return 1;

    return 0;
}

static void normalize_agent_command(char *dst, size_t dst_size, const char *src)
{
    const char *s = src;
    size_t n;

    if (dst_size == 0)
        return;

    if (s == NULL) {
        dst[0] = '\0';
        return;
    }

    while (*s != '\0' && isspace((unsigned char)*s))
        s++;

    if (*s == '`') {
        while (*s == '`')
            s++;
        while (*s != '\0' && isspace((unsigned char)*s))
            s++;
    }

    n = 0;
    while (s[n] != '\0' && s[n] != '\n' && s[n] != '\r' && n + 1 < dst_size) {
        dst[n] = s[n];
        n++;
    }
    dst[n] = '\0';

    if (n > 0 && dst[n - 1] == '`')
        dst[n - 1] = '\0';
}

static int llm_is_safe_parser_command(const char *s)
{
    size_t i;
    size_t len;

    if (s == NULL)
        return 0;

    while (*s != '\0' && isspace((unsigned char)*s))
        s++;
    if (*s == '\0')
        return 0;

    len = strlen(s);
    if (len > 120)
        return 0;

    if (*s == '{' || *s == '[' || *s == '"' || *s == '\'')
        return 0;

    for (i = 0; s[i] != '\0'; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\n' || c == '\r' || c == '{' || c == '}' || c == '[' || c == ']')
            return 0;
    }

    return 1;
}

static char *trim_ws(char *s)
{
    char *end;

    while (*s != '\0' && isspace((unsigned char)*s))
        s++;

    if (*s == '\0')
        return s;

    end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) {
        *end = '\0';
        end--;
    }

    return s;
}

static void assign_if_unset(char **field, const char *value)
{
    if (field == NULL || value == NULL || *value == '\0')
        return;

    if (*field != NULL && **field != '\0')
        return;

    *field = strdup(value);
}

static void strip_optional_quotes(char *value)
{
    size_t len;

    if (value == NULL)
        return;

    len = strlen(value);
    if (len < 2)
        return;

    if ((value[0] == '"' && value[len - 1] == '"') ||
        (value[0] == '\'' && value[len - 1] == '\'')) {
        memmove(value, value + 1, len - 2);
        value[len - 2] = '\0';
    }
}

static int context_lines_setting(void)
{
    const char *env = getenv("FROTZ_LLM_CONTEXT_LINES");
    int n = f_setup.llm_context_lines;

    if (n <= 0 && env != NULL && *env != '\0')
        n = atoi(env);
    if (n <= 0)
        n = 16;
    if (n > 128)
        n = 128;
    return n;
}

static const char *llm_mode_value(void)
{
    return first_nonempty(f_setup.llm_mode, getenv("FROTZ_LLM_MODE"), "off");
}

int llm_mode_is_test(void)
{
    return str_ieq(llm_mode_value(), "test");
}

int llm_mode_is_interactive(void)
{
    return str_ieq(llm_mode_value(), "interactive");
}

static void llm_store_output_line(void)
{
    if (llm_current_line_len == 0)
        return;

    llm_current_line[llm_current_line_len] = '\0';
    strncpy(llm_output_lines[llm_output_next], llm_current_line, LLM_CAPTURE_LINE_LEN - 1);
    llm_output_lines[llm_output_next][LLM_CAPTURE_LINE_LEN - 1] = '\0';
    llm_output_next = (llm_output_next + 1) % LLM_CAPTURE_LINES;
    if (llm_output_count < LLM_CAPTURE_LINES)
        llm_output_count++;
    llm_current_line_len = 0;
    llm_current_line[0] = '\0';
}

static void llm_capture_output_text(const char *s)
{
    size_t i;

    for (i = 0; s[i] != '\0'; i++) {
        unsigned char c = (unsigned char)s[i];

        if (c == '\n') {
            llm_store_output_line();
            continue;
        }
        if (c == '\r')
            continue;
        if (c < 0x20 && c != '\t')
            continue;

        if (llm_current_line_len + 1 >= LLM_CAPTURE_LINE_LEN)
            llm_store_output_line();

        llm_current_line[llm_current_line_len++] = (char)c;
    }
}

void llm_capture_output_word(const unsigned char *s)
{
    llm_capture_output_text((const char *)s);
}

void llm_capture_output_char(unsigned char c)
{
    char tmp[2];
    tmp[0] = (char)c;
    tmp[1] = '\0';
    llm_capture_output_text(tmp);
}

void llm_capture_output_new_line(void)
{
    llm_store_output_line();
}

void llm_capture_user_input(const unsigned char *s, unsigned char terminator)
{
    if (terminator != ZC_RETURN || s == NULL || s[0] == '\0')
        return;

    (void)snprintf(llm_cmd_lines[llm_cmd_next], LLM_CAPTURE_LINE_LEN, "> %s", (const char *)s);
    llm_cmd_next = (llm_cmd_next + 1) % LLM_CAPTURE_CMDS;
    if (llm_cmd_count < LLM_CAPTURE_CMDS)
        llm_cmd_count++;
}

static void llm_append_context(char *dst, size_t dst_size)
{
    int i;
    int max_lines = context_lines_setting();
    int lines = llm_output_count < max_lines ? llm_output_count : max_lines;
    int cmds = llm_cmd_count < (max_lines / 2 + 1) ? llm_cmd_count : (max_lines / 2 + 1);

    strncat(dst, "Recent game output:\n", dst_size - strlen(dst) - 1);
    for (i = 0; i < lines; i++) {
        int idx = llm_output_next - lines + i;
        if (idx < 0)
            idx += LLM_CAPTURE_LINES;
        strncat(dst, llm_output_lines[idx], dst_size - strlen(dst) - 1);
        strncat(dst, "\n", dst_size - strlen(dst) - 1);
    }

    strncat(dst, "\nRecent player commands:\n", dst_size - strlen(dst) - 1);
    for (i = 0; i < cmds; i++) {
        int idx = llm_cmd_next - cmds + i;
        if (idx < 0)
            idx += LLM_CAPTURE_CMDS;
        strncat(dst, llm_cmd_lines[idx], dst_size - strlen(dst) - 1);
        strncat(dst, "\n", dst_size - strlen(dst) - 1);
    }
}

int llm_load_config(const char *path_hint, char *error, size_t error_size)
{
    const char *path;
    const char *fallback_path = NULL;
    const char *home;
    char default_path[PATH_MAX];
    FILE *fp;
    char line[4096];

    if (error == NULL || error_size == 0)
        return 0;

    error[0] = '\0';

    path = first_nonempty(path_hint, getenv("FROTZ_LLM_CONFIG"), NULL);
    if (path == NULL) {
        home = getenv("HOME");
        if (home != NULL && *home != '\0') {
            (void)snprintf(default_path, sizeof(default_path), "%s/.frotz-llm.conf", home);
            path = default_path;
            fallback_path = "frotz-llm.conf";
        }
    }

    if (path == NULL || *path == '\0')
        return 1;

    fp = fopen(path, "r");
    if (fp == NULL) {
        if (errno == ENOENT && fallback_path != NULL) {
            fp = fopen(fallback_path, "r");
            if (fp != NULL)
                path = fallback_path;
        }
    }

    if (fp == NULL) {
        if (errno == ENOENT)
            return 1;
        (void)snprintf(error, error_size, "unable to read config '%s': %s", path, strerror(errno));
        return 0;
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        char *key;
        char *value;
        char *eq;

        key = trim_ws(line);
        if (*key == '\0' || *key == '#' || *key == ';' || *key == '[')
            continue;

        eq = strchr(key, '=');
        if (eq == NULL)
            continue;

        *eq = '\0';
        value = trim_ws(eq + 1);
        key = trim_ws(key);
        strip_optional_quotes(value);

        if (str_ieq(key, "provider") || str_ieq(key, "llm_provider") || str_ieq(key, "FROTZ_LLM_PROVIDER")) {
            assign_if_unset(&f_setup.llm_provider, value);
        } else if (str_ieq(key, "model") || str_ieq(key, "llm_model") || str_ieq(key, "FROTZ_LLM_MODEL")) {
            assign_if_unset(&f_setup.llm_model, value);
        } else if (str_ieq(key, "base_url") || str_ieq(key, "endpoint") || str_ieq(key, "llm_base_url") || str_ieq(key, "FROTZ_LLM_BASE_URL")) {
            assign_if_unset(&f_setup.llm_base_url, value);
        } else if (str_ieq(key, "api_key") || str_ieq(key, "llm_api_key") || str_ieq(key, "FROTZ_LLM_API_KEY")) {
            assign_if_unset(&f_setup.llm_api_key, value);
        } else if (str_ieq(key, "mode") || str_ieq(key, "llm_mode") || str_ieq(key, "FROTZ_LLM_MODE")) {
            assign_if_unset(&f_setup.llm_mode, value);
        } else if (str_ieq(key, "test_prompt") || str_ieq(key, "llm_test_prompt")) {
            assign_if_unset(&f_setup.llm_test_prompt, value);
        } else if (str_ieq(key, "context_lines") || str_ieq(key, "llm_context_lines")) {
            if (f_setup.llm_context_lines <= 0)
                f_setup.llm_context_lines = atoi(value);
        } else if (str_ieq(key, "openrouter_referer") || str_ieq(key, "FROTZ_LLM_REFERER")) {
            if (getenv("FROTZ_LLM_REFERER") == NULL)
                (void)setenv("FROTZ_LLM_REFERER", value, 0);
        } else if (str_ieq(key, "openrouter_title") || str_ieq(key, "FROTZ_LLM_TITLE")) {
            if (getenv("FROTZ_LLM_TITLE") == NULL)
                (void)setenv("FROTZ_LLM_TITLE", value, 0);
        } else if (str_ieq(key, "github_org") || str_ieq(key, "FROTZ_GITHUB_ORG")) {
            if (getenv("FROTZ_GITHUB_ORG") == NULL)
                (void)setenv("FROTZ_GITHUB_ORG", value, 0);
        } else if (str_ieq(key, "github_api_version") || str_ieq(key, "FROTZ_GITHUB_API_VERSION")) {
            if (getenv("FROTZ_GITHUB_API_VERSION") == NULL)
                (void)setenv("FROTZ_GITHUB_API_VERSION", value, 0);
        }
    }

    if (ferror(fp)) {
        fclose(fp);
        (void)snprintf(error, error_size, "error while reading config '%s'", path);
        return 0;
    }

    fclose(fp);
    return 1;
}

int llm_is_configured(void)
{
    const char *provider = first_nonempty(f_setup.llm_provider, getenv("FROTZ_LLM_PROVIDER"), NULL);
    return provider != NULL;
}

int llm_query_with_context(const char *query,
                           char *response,
                           size_t response_size,
                           char *error,
                           size_t error_size)
{
    char prompt[16384];

    if (!llm_mode_is_interactive()) {
        (void)snprintf(error, error_size, "LLM interactive mode is disabled; set mode=interactive");
        return 0;
    }

    prompt[0] = '\0';
    strncat(prompt,
            "You are an assistant helping with a text adventure game. "
            "Use context, avoid spoilers, and keep answers concise.\n\n",
            sizeof(prompt) - strlen(prompt) - 1);
    llm_append_context(prompt, sizeof(prompt));
    strncat(prompt, "\nPlayer request:\n", sizeof(prompt) - strlen(prompt) - 1);
    strncat(prompt, query, sizeof(prompt) - strlen(prompt) - 1);

    return llm_run_prompt(NULL, NULL, NULL, NULL, prompt, response, response_size, error, error_size);
}

int llm_agent_start(const char *goal, char *error, size_t error_size)
{
    const char *p;

    if (error != NULL && error_size > 0)
        error[0] = '\0';

    if (!llm_mode_is_interactive()) {
        if (error != NULL && error_size > 0)
            (void)snprintf(error, error_size, "LLM interactive mode is disabled; set mode=interactive");
        return 0;
    }

    if (goal == NULL) {
        if (error != NULL && error_size > 0)
            (void)snprintf(error, error_size, "missing goal text");
        return 0;
    }

    p = goal;
    while (*p != '\0' && isspace((unsigned char)*p))
        p++;
    if (*p == '\0') {
        if (error != NULL && error_size > 0)
            (void)snprintf(error, error_size, "missing goal text");
        return 0;
    }

    strncpy(llm_agent_goal, goal, sizeof(llm_agent_goal) - 1);
    llm_agent_goal[sizeof(llm_agent_goal) - 1] = '\0';
    llm_agent_step = 0;
    llm_agent_active = 1;
    return 1;
}

void llm_agent_stop(void)
{
    llm_agent_active = 0;
    llm_agent_step = 0;
    llm_agent_goal[0] = '\0';
}

int llm_agent_is_active(void)
{
    return llm_agent_active;
}

int llm_agent_next_command(char *command,
                           size_t command_size,
                           char *status,
                           size_t status_size,
                           char *error,
                           size_t error_size)
{
    char prompt[16384];
    char reply[8192];
    char action[512];
    char summary[2048];
    int done = 0;

    if (command == NULL || command_size == 0 || status == NULL || status_size == 0 || error == NULL || error_size == 0)
        return 0;

    command[0] = '\0';
    status[0] = '\0';
    error[0] = '\0';

    if (!llm_agent_active)
        return 0;

    if (llm_agent_step >= llm_agent_max_steps) {
        (void)snprintf(status, status_size, "Stopped after %d steps. Use /llm do <goal> again to continue.", llm_agent_max_steps);
        llm_agent_stop();
        return 0;
    }

    prompt[0] = '\0';
    strncat(prompt,
            "You are controlling a text adventure parser one command at a time. "
            "Use short parser commands only. Avoid spoilers unless necessary.\n\n",
            sizeof(prompt) - strlen(prompt) - 1);
    strncat(prompt, "Goal:\n", sizeof(prompt) - strlen(prompt) - 1);
    strncat(prompt, llm_agent_goal, sizeof(prompt) - strlen(prompt) - 1);
    strncat(prompt, "\n\n", sizeof(prompt) - strlen(prompt) - 1);
    llm_append_context(prompt, sizeof(prompt));
    strncat(prompt,
            "\nRespond with JSON only in exactly this format:\n"
            "{\"done\":false,\"action\":\"look\",\"summary\":\"\"}\n"
            "When goal is complete or impossible right now, return done=true and put a concise summary in summary.\n",
            sizeof(prompt) - strlen(prompt) - 1);

    if (!llm_run_prompt(NULL, NULL, NULL, NULL, prompt, reply, sizeof(reply), error, error_size)) {
        llm_agent_stop();
        return 0;
    }

    if (parse_agent_json_reply(reply, &done, action, sizeof(action), summary, sizeof(summary))) {
        if (done) {
            if (summary[0] == '\0')
                (void)snprintf(summary, sizeof(summary), "Goal complete.");
            strncpy(status, summary, status_size - 1);
            status[status_size - 1] = '\0';
            llm_agent_stop();
            return 0;
        }

        normalize_agent_command(command, command_size, action);
        if (!llm_is_safe_parser_command(command)) {
            (void)snprintf(error, error_size, "agent returned an invalid action");
            llm_agent_stop();
            return 0;
        }

        llm_agent_step++;
        return 1;
    }

    normalize_agent_command(command, command_size, reply);
    if (command[0] == '\0') {
        (void)snprintf(error, error_size, "agent response did not include a command");
        llm_agent_stop();
        return 0;
    }

    if (command[0] == '{' || command[0] == '[') {
        if (extract_json_string_field_loose(reply, "action", action, sizeof(action))) {
            normalize_agent_command(command, command_size, action);
        }
    }

    if (!llm_is_safe_parser_command(command)) {
        (void)snprintf(error, error_size, "agent response was not a parser command");
        llm_agent_stop();
        return 0;
    }

    if (str_ieq(command, "done") || str_ieq(command, "stop")) {
        (void)snprintf(status, status_size, "Goal complete.");
        llm_agent_stop();
        return 0;
    }

    llm_agent_step++;
    return 1;
}

int llm_run_prompt(const char *provider_hint,
                   const char *model_hint,
                   const char *base_url_hint,
                   const char *api_key_hint,
                   const char *prompt,
                   char *response,
                   size_t response_size,
                   char *error,
                   size_t error_size)
{
    const char *provider_name;
    const char *model;
    const char *endpoint;
    const char *api_key;
    const char *referer;
    const char *title;
    const char *github_org;
    const char *github_api_version;
    llm_provider_t provider;
    char escaped_prompt[8192];
    char payload[12288];
    char payload_path[LLM_PATH_BUF_SIZE];
    char endpoint_buf[1024];
    char cmd[LLM_CMD_BUF_SIZE];
    char body[LLM_RESPONSE_BUF_SIZE];
    int attempt;
    int attempts_used = 0;
    long long t0;

    if (response == NULL || response_size == 0 || error == NULL || error_size == 0)
        return 0;

    response[0] = '\0';
    error[0] = '\0';
    llm_last_latency_ms_value = 0;
    llm_last_attempts_value = 0;
    t0 = llm_now_ms();

    if (prompt == NULL || *prompt == '\0') {
        (void)snprintf(error, error_size, "missing prompt");
        return 0;
    }

    provider_name = first_nonempty(provider_hint, f_setup.llm_provider, getenv("FROTZ_LLM_PROVIDER"));
    provider = llm_provider_from_string(provider_name);
    if (provider == LLM_PROVIDER_NONE) {
        (void)snprintf(error, error_size, "unknown or missing provider; set provider in ~/.frotz-llm.conf or FROTZ_LLM_PROVIDER");
        return 0;
    }

    model = first_nonempty(model_hint, f_setup.llm_model, getenv("FROTZ_LLM_MODEL"));
    if (model == NULL || *model == '\0')
        model = llm_default_model(provider);

    endpoint = first_nonempty(base_url_hint, f_setup.llm_base_url, getenv("FROTZ_LLM_BASE_URL"));
    if (endpoint == NULL || *endpoint == '\0')
        endpoint = llm_default_endpoint(provider);

    api_key = first_nonempty(api_key_hint, f_setup.llm_api_key, getenv("FROTZ_LLM_API_KEY"));
    if ((provider == LLM_PROVIDER_OPENAI || provider == LLM_PROVIDER_OPENROUTER ||
         provider == LLM_PROVIDER_ANTHROPIC || provider == LLM_PROVIDER_GITHUB_MODELS) &&
        (api_key == NULL || *api_key == '\0')) {
        (void)snprintf(error, error_size, "missing API key; set api_key in ~/.frotz-llm.conf or FROTZ_LLM_API_KEY");
        return 0;
    }

    if (provider == LLM_PROVIDER_GITHUB_MODELS) {
        github_org = getenv("FROTZ_GITHUB_ORG");
        if (github_org != NULL && *github_org != '\0') {
            (void)snprintf(endpoint_buf, sizeof(endpoint_buf),
                           "https://models.github.ai/orgs/%s/inference/chat/completions",
                           github_org);
            endpoint = endpoint_buf;
        }
    }

    (void)json_escape(prompt, escaped_prompt, sizeof(escaped_prompt));

    if (provider == LLM_PROVIDER_ANTHROPIC) {
        (void)snprintf(payload, sizeof(payload),
                       "{\"model\":\"%s\",\"max_tokens\":512,\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}]}",
                       model, escaped_prompt);
    } else {
        (void)snprintf(payload, sizeof(payload),
                       "{\"model\":\"%s\",\"temperature\":1,\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}]}",
                       model, escaped_prompt);
    }

    if (!write_temp_payload(payload, payload_path, sizeof(payload_path), error, error_size))
        return 0;

    cmd[0] = '\0';
    strncat(cmd, "curl -sS ", sizeof(cmd) - strlen(cmd) - 1);
    strncat(cmd, "--connect-timeout 10 --max-time 60 ", sizeof(cmd) - strlen(cmd) - 1);
    strncat(cmd, "-X POST ", sizeof(cmd) - strlen(cmd) - 1);
    shell_append_quoted(cmd, sizeof(cmd), endpoint);
    shell_append_header(cmd, sizeof(cmd), "Content-Type", "application/json");

    if (provider == LLM_PROVIDER_ANTHROPIC) {
        shell_append_header(cmd, sizeof(cmd), "anthropic-version", "2023-06-01");
        shell_append_header(cmd, sizeof(cmd), "x-api-key", api_key);
    } else if (api_key != NULL && *api_key != '\0') {
        char bearer[1024];
        (void)snprintf(bearer, sizeof(bearer), "Bearer %s", api_key);
        shell_append_header(cmd, sizeof(cmd), "Authorization", bearer);
    }

    if (provider == LLM_PROVIDER_GITHUB_MODELS) {
        github_api_version = getenv("FROTZ_GITHUB_API_VERSION");
        shell_append_header(cmd, sizeof(cmd), "Accept", "application/vnd.github+json");
        shell_append_header(cmd, sizeof(cmd), "X-GitHub-Api-Version",
                            (github_api_version != NULL && *github_api_version != '\0') ? github_api_version : "2026-03-10");
    }

    if (provider == LLM_PROVIDER_OPENROUTER) {
        referer = getenv("FROTZ_LLM_REFERER");
        title = getenv("FROTZ_LLM_TITLE");
        if (referer != NULL && *referer != '\0')
            shell_append_header(cmd, sizeof(cmd), "HTTP-Referer", referer);
        if (title != NULL && *title != '\0')
            shell_append_header(cmd, sizeof(cmd), "X-Title", title);
    }

    strncat(cmd, " --data-binary @", sizeof(cmd) - strlen(cmd) - 1);
    shell_append_quoted(cmd, sizeof(cmd), payload_path);
    strncat(cmd, " 2>&1", sizeof(cmd) - strlen(cmd) - 1);

    llm_logf(1, "request start provider=%s model=%s endpoint=%s prompt_chars=%lu",
             provider_name != NULL ? provider_name : "(unset)",
             model != NULL ? model : "(default)",
             endpoint != NULL ? endpoint : "(unset)",
             (unsigned long)strlen(prompt));
    llm_log_block(2, "PROMPT", prompt);
    llm_log_block(2, "JSON PAYLOAD", payload);

    for (attempt = 0; attempt < 2; attempt++) {
        long long t_attempt = llm_now_ms();
        attempts_used = attempt + 1;
        llm_logf(1, "attempt %d/2: sending request", attempt + 1);

        if (!run_curl_command(cmd, body, sizeof(body), error, error_size)) {
            llm_last_latency_ms_value = (long)(llm_now_ms() - t0);
            llm_last_attempts_value = attempts_used;
            llm_logf(1, "attempt %d/2 failed after %lld ms: %s",
                     attempt + 1,
                     (long long)(llm_now_ms() - t_attempt),
                     error);
            unlink(payload_path);
            return 0;
        }

        llm_logf(1, "attempt %d/2 received %lu bytes in %lld ms",
                 attempt + 1,
                 (unsigned long)strlen(body),
                 (long long)(llm_now_ms() - t_attempt));
        llm_log_block(2, "JSON RESPONSE", body);

        if (extract_json_response_text(body, response, response_size)) {
            llm_last_latency_ms_value = (long)(llm_now_ms() - t0);
            llm_last_attempts_value = attempts_used;
            llm_logf(1, "request complete in %ld ms (%d attempt%s)",
                     llm_last_latency_ms_value,
                     llm_last_attempts_value,
                     llm_last_attempts_value == 1 ? "" : "s");
            unlink(payload_path);
            return 1;
        }

        if (llm_verbose_level() >= 1) {
            llm_logf(1, "attempt %d parse failed, retrying", attempt + 1);
        }
    }

    llm_last_latency_ms_value = (long)(llm_now_ms() - t0);
    llm_last_attempts_value = attempts_used;
    unlink(payload_path);
    (void)snprintf(error, error_size, "could not parse LLM response: %s", body);
    return 0;
}

long llm_last_latency_ms(void)
{
    return llm_last_latency_ms_value;
}

int llm_last_attempts(void)
{
    return llm_last_attempts_value;
}

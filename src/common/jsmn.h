/*
 * JSON parser from https://github.com/zserge/jsmn
 * MIT licensed, single-header copy for embedded use.
 */

#ifndef JSMN_H
#define JSMN_H

#include <stddef.h>

typedef enum {
    JSMN_UNDEFINED = 0,
    JSMN_OBJECT = 1,
    JSMN_ARRAY = 2,
    JSMN_STRING = 3,
    JSMN_PRIMITIVE = 4
} jsmntype_t;

enum jsmnerr {
    JSMN_ERROR_NOMEM = -1,
    JSMN_ERROR_INVAL = -2,
    JSMN_ERROR_PART = -3
};

typedef struct {
    jsmntype_t type;
    int start;
    int end;
    int size;
#ifdef JSMN_PARENT_LINKS
    int parent;
#endif
} jsmntok_t;

typedef struct {
    unsigned int pos;
    unsigned int toknext;
    int toksuper;
} jsmn_parser;

static void jsmn_init(jsmn_parser *parser)
{
    parser->pos = 0;
    parser->toknext = 0;
    parser->toksuper = -1;
}

static jsmntok_t *jsmn_alloc_token(jsmn_parser *parser, jsmntok_t *tokens, size_t num_tokens)
{
    jsmntok_t *tok;
    if (parser->toknext >= num_tokens)
        return NULL;
    tok = &tokens[parser->toknext++];
    tok->start = tok->end = -1;
    tok->size = 0;
#ifdef JSMN_PARENT_LINKS
    tok->parent = -1;
#endif
    return tok;
}

static void jsmn_fill_token(jsmntok_t *token, jsmntype_t type, int start, int end)
{
    token->type = type;
    token->start = start;
    token->end = end;
    token->size = 0;
}

static int jsmn_parse_primitive(jsmn_parser *parser, const char *js, size_t len,
                                jsmntok_t *tokens, size_t num_tokens)
{
    int start = parser->pos;

    for (; parser->pos < len; parser->pos++) {
        char c = js[parser->pos];
        if (c == ':' || c == '\t' || c == '\r' || c == '\n' || c == ' ' ||
            c == ',' || c == ']' || c == '}') {
            break;
        }
        if (c < 32 || c >= 127) {
            parser->pos = start;
            return JSMN_ERROR_INVAL;
        }
    }

    if (tokens == NULL) {
        parser->pos--;
        return 0;
    }

    {
        jsmntok_t *token = jsmn_alloc_token(parser, tokens, num_tokens);
        if (token == NULL) {
            parser->pos = start;
            return JSMN_ERROR_NOMEM;
        }
        jsmn_fill_token(token, JSMN_PRIMITIVE, start, parser->pos);
#ifdef JSMN_PARENT_LINKS
        token->parent = parser->toksuper;
#endif
    }
    parser->pos--;
    return 0;
}

static int jsmn_parse_string(jsmn_parser *parser, const char *js, size_t len,
                             jsmntok_t *tokens, size_t num_tokens)
{
    int start = parser->pos;

    parser->pos++;

    for (; parser->pos < len; parser->pos++) {
        char c = js[parser->pos];

        if (c == '"') {
            if (tokens == NULL)
                return 0;
            {
                jsmntok_t *token = jsmn_alloc_token(parser, tokens, num_tokens);
                if (token == NULL) {
                    parser->pos = start;
                    return JSMN_ERROR_NOMEM;
                }
                jsmn_fill_token(token, JSMN_STRING, start + 1, parser->pos);
#ifdef JSMN_PARENT_LINKS
                token->parent = parser->toksuper;
#endif
            }
            return 0;
        }

        if (c == '\\' && parser->pos + 1 < len) {
            parser->pos++;
            switch (js[parser->pos]) {
            case '"':
            case '/':
            case '\\':
            case 'b':
            case 'f':
            case 'r':
            case 'n':
            case 't':
                break;
            case 'u':
                parser->pos++;
                for (int i = 0; i < 4 && parser->pos < len; i++, parser->pos++) {
                    char hc = js[parser->pos];
                    if (!((hc >= '0' && hc <= '9') ||
                          (hc >= 'A' && hc <= 'F') ||
                          (hc >= 'a' && hc <= 'f'))) {
                        parser->pos = start;
                        return JSMN_ERROR_INVAL;
                    }
                }
                parser->pos--;
                break;
            default:
                parser->pos = start;
                return JSMN_ERROR_INVAL;
            }
        }
    }

    parser->pos = start;
    return JSMN_ERROR_PART;
}

static int jsmn_parse(jsmn_parser *parser, const char *js, size_t len,
                      jsmntok_t *tokens, unsigned int num_tokens)
{
    int r;
    int i;
    jsmntok_t *token;

    for (; parser->pos < len; parser->pos++) {
        char c = js[parser->pos];
        switch (c) {
        case '{':
        case '[':
            token = jsmn_alloc_token(parser, tokens, num_tokens);
            if (token == NULL)
                return JSMN_ERROR_NOMEM;
            if (parser->toksuper != -1)
                tokens[parser->toksuper].size++;
            token->type = (c == '{' ? JSMN_OBJECT : JSMN_ARRAY);
            token->start = parser->pos;
#ifdef JSMN_PARENT_LINKS
            token->parent = parser->toksuper;
#endif
            parser->toksuper = (int)(parser->toknext - 1);
            break;
        case '}':
        case ']':
            if (tokens == NULL)
                break;
            for (i = (int)parser->toknext - 1; i >= 0; i--) {
                token = &tokens[i];
                if (token->start != -1 && token->end == -1) {
                    if ((token->type == JSMN_OBJECT && c == ']') ||
                        (token->type == JSMN_ARRAY && c == '}')) {
                        return JSMN_ERROR_INVAL;
                    }
                    token->end = parser->pos + 1;
                    parser->toksuper = -1;
#ifdef JSMN_PARENT_LINKS
                    parser->toksuper = token->parent;
#else
                    for (i--; i >= 0; i--) {
                        if (tokens[i].start != -1 && tokens[i].end == -1) {
                            parser->toksuper = i;
                            break;
                        }
                    }
#endif
                    break;
                }
            }
            if (i == -1)
                return JSMN_ERROR_INVAL;
            break;
        case '"':
            r = jsmn_parse_string(parser, js, len, tokens, num_tokens);
            if (r < 0)
                return r;
            if (parser->toksuper != -1)
                tokens[parser->toksuper].size++;
            break;
        case '\t':
        case '\r':
        case '\n':
        case ' ':
        case ':':
        case ',':
            break;
        default:
            r = jsmn_parse_primitive(parser, js, len, tokens, num_tokens);
            if (r < 0)
                return r;
            if (parser->toksuper != -1)
                tokens[parser->toksuper].size++;
            break;
        }
    }

    for (i = (int)parser->toknext - 1; i >= 0; i--) {
        if (tokens[i].start != -1 && tokens[i].end == -1)
            return JSMN_ERROR_PART;
    }

    return (int)parser->toknext;
}

#endif

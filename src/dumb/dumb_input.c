/*
 * dumb_input.c - Dumb interface, input functions
 *
 * This file is part of Frotz.
 *
 * Frotz is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Frotz is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 * Or visit http://www.fsf.org/
 */

#include <string.h>
#include <ctype.h>

#include "dumb_frotz.h"
#include "../common/llm.h"

f_setup_t f_setup;

static char runtime_usage[] =
  "DUMB-FROTZ runtime help:\n"
  "  General Commands:\n"
  "    /help            Show this message.\n"
  "    /set             Show current settings.\n"
  "    /ask <text>      Ask the LLM using recent game context.\n"
  "    /hint            Ask for one concise, low-spoiler hint.\n"
  "    /next [goal]     Suggest one next parser command (no autopilot).\n"
  "    /play N          Iteratively execute next N moves.\n"
  "    /stop            Stop autopilot.\n"
  "    /status          Show autopilot status.\n"
  "    /s               Show the current contents of the whole screen.\n"
  "    /d               Discard the part of the input before the cursor.\n"
  "    /wN              Advance clock N/10 seconds, possibly causing the current\n"
  "                        and subsequent inputs to timeout.\n"
  "    /w               Advance clock by the amount of real time since this input\n"
  "                        started (times the current speed factor).\n"
  "    /t               Advance clock just enough to timeout the current input\n"
  "  Reverse-Video Display Method Settings:\n"
  "    /rn   none    /rc   CAPS    /rd   doublestrike    /ru   underline\n"
  "    /rbC  show rv blanks as char C (orthogonal to above modes)\n"
  "  Output Compression Settings:\n"
  "    /cn      none: show whole screen before every input.\n"
  "    /cm      max: show only lines that have new nonblank characters.\n"
  "    /cs      spans: like max, but emit a blank line between each span of\n"
  "                screen lines shown.\n"
  "    /chN     Hide top N lines (orthogonal to above modes).\n"
  "  Misc Settings:\n"
  "    /sfX     Set speed factor to X.  (0 = never timeout automatically).\n"
  "    /mp      Toggle use of MORE prompts\n"
  "    /ln      Toggle display of line numbers.\n"
  "    /lt      Toggle display of the line type identification chars.\n"
  "    /vb      Toggle visual bell.\n"
  "    /pb      Toggle display of picture outline boxes.\n"
  "    (Toggle commands can be followed by a 1 or 0 to set value ON or OFF.)\n"
  "  Character Escapes:\n"
  "    \\\\  backslash    \\#  backspace    \\[  escape    \\_  return\n"
  "    \\< \\> \\^ \\.  cursor motion        \\1 ..\\0  f1..f10\n"
  "    \\D ..\\X   Standard Frotz hotkeys.  Use \\H (help) to see the list.\n"
  "  Line Type Identification Characters:\n"
  "    Input lines:\n"
  "      untimed  timed\n"
  "      >        T      A regular line-oriented input\n"
  "      )        t      A single-character input\n"
  "      }        D      A line input with some input before the cursor.\n"
  "                         (Use /d to discard it.)\n"
  "    Output lines:\n"
  "      ]     Output line that contains the cursor.\n"
  "      .     A blank line emitted as part of span compression.\n"
  "            (blank) Any other output line.\n"
;

static float speed = 1;

enum input_type {
    INPUT_CHAR,
    INPUT_LINE,
    INPUT_LINE_CONTINUED,
};

/* get a character.  Exit with no fuss on EOF.  */
static int xgetchar(void)
{
    int c = getchar();
    if (c == EOF) {
	if (feof(stdin)) {
	    fprintf(stderr, "\nEOT\n");
	    exit(0);
	}
	os_fatal(strerror(errno));
    }
    return c;
}

/* Read one line, including the newline, into s.  Safely avoids buffer
 * overruns (but that's kind of pointless because there are several
 * other places where I'm not so careful).  */
static void dumb_getline(char *s)
{
    int c;
    char *p = s;
    while (p < s + INPUT_BUFFER_SIZE - 1) {
	if ((*p++ = xgetchar()) == '\n') {
	    *p = '\0';
	    return;
	}
    }

    p[-1] = '\n';
    p[0] = '\0';
    while ((c = xgetchar()) != '\n')
 	;
    printf("Line too long, truncated to %s\n", s - INPUT_BUFFER_SIZE);
}

/* Translate in place all the escape characters in s.  */
static void translate_special_chars(char *s)
{
  char *src = s, *dest = s;
  while (*src)
    switch(*src++) {
    default: *dest++ = src[-1]; break;
    case '\n': *dest++ = ZC_RETURN; break;
    case '\\':
      switch (*src++) {
      case '\n': *dest++ = ZC_RETURN; break;
      case '\\': *dest++ = '\\'; break;
      case '?': *dest++ = ZC_BACKSPACE; break;
      case '[': *dest++ = ZC_ESCAPE; break;
      case '_': *dest++ = ZC_RETURN; break;
      case '^': *dest++ = ZC_ARROW_UP; break;
      case '.': *dest++ = ZC_ARROW_DOWN; break;
      case '<': *dest++ = ZC_ARROW_LEFT; break;
      case '>': *dest++ = ZC_ARROW_RIGHT; break;
      case 'R': *dest++ = ZC_HKEY_RECORD; break;
      case 'P': *dest++ = ZC_HKEY_PLAYBACK; break;
      case 'S': *dest++ = ZC_HKEY_SEED; break;
      case 'U': *dest++ = ZC_HKEY_UNDO; break;
      case 'N': *dest++ = ZC_HKEY_RESTART; break;
      case 'X': *dest++ = ZC_HKEY_QUIT; break;
      case 'D': *dest++ = ZC_HKEY_DEBUG; break;
      case 'H': *dest++ = ZC_HKEY_HELP; break;
      case '1': case '2': case '3': case '4':
      case '5': case '6': case '7': case '8': case '9':
	*dest++ = ZC_FKEY_MIN + src[-1] - '0' - 1; break;
      case '0': *dest++ = ZC_FKEY_MIN + 9; break;
      default:
	fprintf(stderr, "DUMB-FROTZ: unknown escape char: %c\n", src[-1]);
  fprintf(stderr, "Enter /help to see the list\n");
      }
    }
  *dest = '\0';
}


/* The time in tenths of seconds that the user is ahead of z time.  */
static int time_ahead = 0;

/* Called from os_read_key and os_read_line if they have input from
 * a previous call to dumb_read_line.
 * Returns TRUE if we should timeout rather than use the read-ahead.
 * (because the user is further ahead than the timeout).  */
static bool check_timeout(int timeout)
{
    if ((timeout == 0) || (timeout > time_ahead))
	time_ahead = 0;
    else
	time_ahead -= timeout;
    return time_ahead != 0;
}

/* If val is '0' or '1', set *var accordingly, otherwise toggle it.  */
static void toggle(bool *var, char val)
{
    *var = val == '1' || (val != '0' && !*var);
}

static bool llm_command_looks_json(const char *s)
{
  while (s != NULL && *s != '\0' && isspace((unsigned char)*s))
    s++;

  return s != NULL && (*s == '{' || *s == '[');
}

static bool llm_extract_next_command(const char *response, char *out, size_t out_size)
{
  const char *p;
  const char *end;
  size_t n;

  if (response == NULL || out == NULL || out_size == 0)
    return FALSE;

  p = response;
  while (*p != '\0' && isspace((unsigned char)*p))
    p++;

  if (p[0] == '`' && p[1] == '`' && p[2] == '`') {
    while (*p != '\0' && *p != '\n')
      p++;
    while (*p == '\n' || *p == '\r')
      p++;
  }

  end = p;
  while (*end != '\0' && *end != '\n' && *end != '\r')
    end++;
  while (end > p && isspace((unsigned char)end[-1]))
    end--;

  if (end <= p)
    return FALSE;

  n = (size_t)(end - p);
  if (n >= out_size)
    n = out_size - 1;
  memcpy(out, p, n);
  out[n] = '\0';

  if (llm_command_looks_json(out))
    return FALSE;

  return out[0] != '\0';
}

static int llm_play_remaining = 0;

static bool llm_next_command_query(char *out_cmd, size_t out_size, char *error, size_t error_size)
{
  const char *query = "Based on context, suggest exactly one safe next parser command. Return only the command text.";
  char response[8192];

  if (!llm_query_with_context(query, response, sizeof(response), error, error_size))
    return FALSE;

  if (!llm_extract_next_command(response, out_cmd, out_size)) {
    (void)snprintf(error, error_size, "couldn't extract a parser command from LLM response");
    return FALSE;
  }

  return TRUE;
}

/* Handle input-related user settings and call dumb_output_handle_setting.  */
bool dumb_handle_setting(const char *setting, bool show_cursor, bool startup)
{
    if (!strncmp(setting, "sf", 2)) {
	speed = atof(&setting[2]);
	printf("Speed Factor %g\n", speed);
    } else if (!strncmp(setting, "mp", 2)) {
	toggle(&do_more_prompts, setting[2]);
	printf("More prompts %s\n", do_more_prompts ? "ON" : "OFF");
    } else {
	if (!strcmp(setting, "set")) {
	    printf("Speed Factor %g\n", speed);
	    printf("More Prompts %s\n", do_more_prompts ? "ON" : "OFF");
	}
	return dumb_output_handle_setting(setting, show_cursor, startup);
    }
    return TRUE;
}

/* Read a line, processing commands (lines that start with a slash
 * (that isn't the start of a special character)), and write the
 * first non-command to s.
 * Return true if timed-out.  */
static bool dumb_read_line(char *s, char *prompt, bool show_cursor,
			   int timeout, enum input_type type,
			   zchar *continued_line_chars)
{
  time_t start_time;

  if (timeout) {
    if (time_ahead >= timeout) {
      time_ahead -= timeout;
      return TRUE;
    }
    timeout -= time_ahead;
    start_time = time(0);
  }
  time_ahead = 0;

  dumb_show_screen(show_cursor);
  for (;;) {
    if (llm_agent_is_active()) {
      char auto_cmd[INPUT_BUFFER_SIZE];
      char status[2048];
      char error[4096];

      if (llm_agent_next_command(auto_cmd, sizeof(auto_cmd), status, sizeof(status), error, sizeof(error))) {
  if (llm_command_looks_json(auto_cmd)) {
    llm_agent_stop();
    fprintf(stdout, "LLM agent error: rejected non-parser action: %s\n", auto_cmd);
    continue;
  }
	fprintf(stdout, "[LLM->game] %s\n", auto_cmd);
      {
        size_t n = strnlen(auto_cmd, INPUT_BUFFER_SIZE - 2);
        memcpy(s, auto_cmd, n);
        s[n++] = '\n';
        s[n] = '\0';
      }
	return FALSE;
      }

	if (status[0] != '\0')
	  fprintf(stdout, "[LLM] %s\n", status);
	if (error[0] != '\0')
	  fprintf(stdout, "LLM agent error: %s\n", error);
    }

    if (llm_play_remaining > 0) {
      char play_cmd[INPUT_BUFFER_SIZE];
      char play_error[4096];

      if (!llm_mode_is_interactive()) {
        llm_play_remaining = 0;
        fputs("[LLM] /play stopped: interactive mode disabled.\n", stdout);
      } else {
        fputs("[LLM] Waiting for response...\n", stdout);
        if (!llm_next_command_query(play_cmd, sizeof(play_cmd), play_error, sizeof(play_error))) {
          llm_play_remaining = 0;
          fprintf(stdout, "LLM error: %s\n", play_error);
        } else {
          size_t n;
          llm_play_remaining--;
          fprintf(stdout, "[LLM->game] %s (%d move%s left)\n",
                  play_cmd,
                  llm_play_remaining,
                  llm_play_remaining == 1 ? "" : "s");
          n = strnlen(play_cmd, INPUT_BUFFER_SIZE - 2);
          memcpy(s, play_cmd, n);
          s[n++] = '\n';
          s[n] = '\0';
          return FALSE;
        }
      }
    }

    char *command;
    if (prompt)
      fputs(prompt, stdout);
    else
      dumb_show_prompt(show_cursor, (timeout ? "tTD" : ")>}")[type]);
    /* Prompt only shows up after user input if we don't flush stdout */
    fflush(stdout);
    dumb_getline(s);
    if ((s[0] != '/') || ((s[1] != '\0') && (s[1] != '\n') && !islower((unsigned char)s[1]))) {
      /* Is not a command line.  */
      translate_special_chars(s);
      if (timeout) {
	int elapsed = (time(0) - start_time) * 10 * speed;
	if (elapsed > timeout) {
	  time_ahead = elapsed - timeout;
	  return TRUE;
	}
      }
      return FALSE;
    }
    /* Commands.  */

    /* Remove the / and the terminating newline.  */
    command = s + 1;
    command[strlen(command) - 1] = '\0';

    if (!strcmp(command, "")) {
      fputs("\nAvailable commands:\n", stdout);
      fputs("  /help           Show this list\n", stdout);
      fputs("  /set            Show current settings\n", stdout);
      fputs("  /ask <text>     Ask the LLM with game context\n", stdout);
      fputs("  /hint           Ask for a spoiler-light hint\n", stdout);
      fputs("  /next [goal]    Suggest one next parser command\n", stdout);
      fputs("  /play N         Iteratively execute next N moves\n", stdout);
      fputs("  /stop           Stop autopilot\n", stdout);
      fputs("  /status         Show autopilot status\n", stdout);
    } else if (!strcmp(command, "t")) {
      if (timeout) {
	time_ahead = 0;
	s[0] = '\0';
	return TRUE;
      }
    } else if (*command == 'w') {
      if (timeout) {
	int elapsed = atoi(&command[1]);
	time_t now = time(0);
	if (elapsed == 0)
	  elapsed = (now - start_time) * 10 * speed;
	if (elapsed >= timeout) {
	  time_ahead = elapsed - timeout;
	  s[0] = '\0';
	  return TRUE;
	}
	timeout -= elapsed;
	start_time = now;
      }
    } else if (!strcmp(command, "d")) {
      if (type != INPUT_LINE_CONTINUED)
	fprintf(stderr, "DUMB-FROTZ: No input to discard\n");
      else {
	dumb_discard_old_input(strlen((char*) continued_line_chars));
	continued_line_chars[0] = '\0';
	type = INPUT_LINE;
      }
    } else if (!strcmp(command, "help")) {
      if (!do_more_prompts)
	fputs(runtime_usage, stdout);
      else {
	char *current_page, *next_page;
	current_page = next_page = runtime_usage;
	for (;;) {
	  int i;
	  for (i = 0; (i < h_screen_rows - 2) && *next_page; i++)
	    next_page = strchr(next_page, '\n') + 1;
	  /* next_page - current_page is width */
	  printf("%.*s", (int) (next_page - current_page), current_page);
	  current_page = next_page;
	  if (!*current_page)
	    break;
	  printf("HELP: Type <return> for more, or q <return> to stop: ");
	  fflush(stdout);
	  dumb_getline(s);
	  if (!strcmp(s, "q\n"))
	    break;
	}
      }
    } else if (!strcmp(command, "set")) {
      const char *provider = f_setup.llm_provider && *f_setup.llm_provider ? f_setup.llm_provider : "(unset)";
      const char *model    = f_setup.llm_model    && *f_setup.llm_model    ? f_setup.llm_model    : "(default)";
      const char *mode     = f_setup.llm_mode     && *f_setup.llm_mode     ? f_setup.llm_mode     : "off";
      fprintf(stdout, "Settings:\n  provider:      %s\n  model:         %s\n  mode:          %s\n  context_lines: %d\n",
              provider, model, mode,
              f_setup.llm_context_lines > 0 ? f_setup.llm_context_lines : 16);
        } else if (!strncmp(command, "ask", 3) && (command[3] == '\0' || isspace((unsigned char)command[3]))) {
      const char *query;
      char response[8192];
      char error[4096];
      char *args = command + 3;
      while (*args != '\0' && isspace((unsigned char)*args)) args++;
      if (!llm_mode_is_interactive()) { fputs("LLM interactive mode disabled.\n", stdout); continue; }
      query = (*args == '\0') ? "Give one concise suggestion for what to try next." : args;
          fputs("[LLM] Waiting for response...\n", stdout);
      if (!llm_query_with_context(query, response, sizeof(response), error, sizeof(error)))
        fprintf(stdout, "LLM error: %s\n", error);
      else
            fprintf(stdout, "[LLM] (%ld ms, %d attempt%s) %s\n",
                    llm_last_latency_ms(),
                    llm_last_attempts(),
                    llm_last_attempts() == 1 ? "" : "s",
                    response);
    } else if (!strncmp(command, "hint", 4) && (command[4] == '\0' || isspace((unsigned char)command[4]))) {
      char response[8192];
      char error[4096];
      if (!llm_mode_is_interactive()) { fputs("LLM interactive mode disabled.\n", stdout); continue; }
          fputs("[LLM] Waiting for response...\n", stdout);
      if (!llm_query_with_context("Give exactly one concise hint based on context. Avoid major spoilers.", response, sizeof(response), error, sizeof(error)))
        fprintf(stdout, "LLM error: %s\n", error);
      else
            fprintf(stdout, "[LLM] (%ld ms, %d attempt%s) %s\n",
                    llm_last_latency_ms(),
                    llm_last_attempts(),
                    llm_last_attempts() == 1 ? "" : "s",
                    response);
    } else if (!strncmp(command, "next", 4) && (command[4] == '\0' || isspace((unsigned char)command[4]))) {
      const char *query;
      char response[8192];
      char next_cmd[INPUT_BUFFER_SIZE];
      char error[4096];
      char querybuf[1024];
      char *goal = command + 4;
      while (*goal != '\0' && isspace((unsigned char)*goal)) goal++;
      if (!llm_mode_is_interactive()) { fputs("LLM interactive mode disabled.\n", stdout); continue; }
      if (*goal == '\0')
        query = "Based on context, suggest exactly one safe next parser command. Return only the command text.";
      else {
        (void)snprintf(querybuf, sizeof(querybuf),
                       "Goal: %s\\nSuggest exactly one safe next parser command. Return only the command text.",
                       goal);
        query = querybuf;
      }
      fputs("[LLM] Waiting for response...\n", stdout);
      if (!llm_query_with_context(query, response, sizeof(response), error, sizeof(error)))
        fprintf(stdout, "LLM error: %s\n", error);
      else {
        fprintf(stdout, "[LLM] (%ld ms, %d attempt%s) %s\n",
                llm_last_latency_ms(),
                llm_last_attempts(),
                llm_last_attempts() == 1 ? "" : "s",
                response);
        if (!llm_extract_next_command(response, next_cmd, sizeof(next_cmd))) {
          fputs("LLM error: couldn't extract a parser command from /next response.\n", stdout);
          continue;
        }
        fprintf(stdout, "[LLM->game] %s\n", next_cmd);
        {
          size_t n = strnlen(next_cmd, INPUT_BUFFER_SIZE - 2);
          memcpy(s, next_cmd, n);
          s[n++] = '\n';
          s[n] = '\0';
        }
        return FALSE;
      }
          } else if (!strncmp(command, "play", 4) && (command[4] == '\0' || isspace((unsigned char)command[4]))) {
            char *args = command + 4;
            char *endptr = NULL;
            long n;
            while (*args != '\0' && isspace((unsigned char)*args)) args++;
            if (!llm_mode_is_interactive()) { fputs("LLM interactive mode disabled.\n", stdout); continue; }
            if (*args == '\0') { fputs("Usage: /play N\n", stdout); continue; }
            n = strtol(args, &endptr, 10);
            while (endptr != NULL && *endptr != '\0' && isspace((unsigned char)*endptr)) endptr++;
            if (endptr == args || (endptr != NULL && *endptr != '\0') || n <= 0 || n > 1000) {
              fputs("Usage: /play N (1..1000)\n", stdout);
              continue;
            }
            llm_play_remaining = (int)n;
            fprintf(stdout, "[LLM] Queued %d move%s via /play.\n",
                    llm_play_remaining,
                    llm_play_remaining == 1 ? "" : "s");
    } else if (!strncmp(command, "do", 2) && (command[2] == '\0' || isspace((unsigned char)command[2]))) {
      fputs("[LLM] /do is temporarily disabled. Use /next [goal] for one-step guidance.\n", stdout);
    } else if (!strcmp(command, "stop")) {
      llm_agent_stop();
            llm_play_remaining = 0;
      fputs("[LLM] Autopilot stopped.\n", stdout);
    } else if (!strcmp(command, "status")) {
            if (llm_play_remaining > 0)
              fprintf(stdout, "[LLM] /play has %d move%s remaining.\n",
                      llm_play_remaining,
                      llm_play_remaining == 1 ? "" : "s");
      fprintf(stdout, "[LLM] Autopilot is %s.\n", llm_agent_is_active() ? "active" : "inactive");
    } else if (!strcmp(command, "s")) {
	dumb_dump_screen();
    } else if (!dumb_handle_setting(command, show_cursor, FALSE)) {
      const char *query;
      char response[8192];
      char next_cmd[INPUT_BUFFER_SIZE];
      char error[4096];
      char querybuf[1024];
      char *goal = command;
      while (*goal != '\0' && isspace((unsigned char)*goal)) goal++;
      if (!llm_mode_is_interactive()) { fputs("LLM interactive mode disabled.\n", stdout); continue; }
      if (*goal == '\0')
        query = "Based on context, suggest exactly one safe next parser command. Return only the command text.";
      else {
        (void)snprintf(querybuf, sizeof(querybuf),
                       "Goal: %s\\nSuggest exactly one safe next parser command. Return only the command text.",
                       goal);
        query = querybuf;
      }
      fprintf(stdout, "[LLM] Treating /%s as /next %s\n", command, goal);
      fputs("[LLM] Waiting for response...\n", stdout);
      if (!llm_query_with_context(query, response, sizeof(response), error, sizeof(error))) {
        fprintf(stdout, "LLM error: %s\n", error);
        continue;
      }
      fprintf(stdout, "[LLM] (%ld ms, %d attempt%s) %s\n",
              llm_last_latency_ms(),
              llm_last_attempts(),
              llm_last_attempts() == 1 ? "" : "s",
              response);
      if (!llm_extract_next_command(response, next_cmd, sizeof(next_cmd))) {
        fputs("LLM error: couldn't extract a parser command from /next response.\n", stdout);
        continue;
      }
      fprintf(stdout, "[LLM->game] %s\n", next_cmd);
      {
        size_t n = strnlen(next_cmd, INPUT_BUFFER_SIZE - 2);
        memcpy(s, next_cmd, n);
        s[n++] = '\n';
        s[n] = '\0';
      }
      return FALSE;
    }
  }
}

/* Read a line that is not part of z-machine input (more prompts and
 * filename requests).  */
static void dumb_read_misc_line(char *s, char *prompt)
{
  dumb_read_line(s, prompt, 0, 0, 0, 0);
  /* Remove terminating newline */
  s[strlen(s) - 1] = '\0';
}

/* For allowing the user to input in a single line keys to be returned
 * for several consecutive calls to read_char, with no screen update
 * in between.  Useful for traversing menus.  */
static char read_key_buffer[INPUT_BUFFER_SIZE];

/* Similar.  Useful for using function key abbreviations.  */
static char read_line_buffer[INPUT_BUFFER_SIZE];

zchar os_read_key (int timeout, bool show_cursor)
{
  char c;
  int timed_out;

  /* Discard any keys read for line input.  */
  read_line_buffer[0] = '\0';

  if (read_key_buffer[0] == '\0') {
    timed_out = dumb_read_line(read_key_buffer, NULL, show_cursor, timeout,
			       INPUT_CHAR, NULL);
    /* An empty input line is reported as a single CR.
     * If there's anything else in the line, we report only the line's
     * contents and not the terminating CR.  */
    if (strlen(read_key_buffer) > 1)
      read_key_buffer[strlen(read_key_buffer) - 1] = '\0';
  } else
    timed_out = check_timeout(timeout);

  if (timed_out)
    return ZC_TIME_OUT;

  c = read_key_buffer[0];
  memmove(read_key_buffer, read_key_buffer + 1, strlen(read_key_buffer));

  /* TODO: error messages for invalid special chars.  */

  return c;
}

zchar os_read_line (int UNUSED (max), zchar *buf, int timeout, int UNUSED(width), int continued)
{
  char *p;
  int terminator;
  static bool timed_out_last_time;
  int timed_out;

  /* Discard any keys read for single key input.  */
  read_key_buffer[0] = '\0';

  /* After timing out, discard any further input unless we're continuing.  */
  if (timed_out_last_time && !continued)
    read_line_buffer[0] = '\0';

  if (read_line_buffer[0] == '\0')
    timed_out = dumb_read_line(read_line_buffer, NULL, TRUE, timeout,
			       buf[0] ? INPUT_LINE_CONTINUED : INPUT_LINE,
			       buf);
  else
    timed_out = check_timeout(timeout);

  if (timed_out) {
    timed_out_last_time = TRUE;
    return ZC_TIME_OUT;
  }

  /* find the terminating character.  */
  for (p = read_line_buffer;; p++) {
    if (is_terminator(*p)) {
      terminator = *p;
      *p++ = '\0';
      break;
    }
  }

  /* TODO: Truncate to width and max.  */

  /* copy to screen */
  dumb_display_user_input(read_line_buffer);

  /* copy to the buffer and save the rest for next time.  */
  strcat((char*) buf, read_line_buffer);
  p = read_line_buffer + strlen(read_line_buffer) + 1;
  memmove(read_line_buffer, p, strlen(p) + 1);

  /* If there was just a newline after the terminating character,
   * don't save it.  */
  if ((read_line_buffer[0] == '\r') && (read_line_buffer[1] == '\0'))
    read_line_buffer[0] = '\0';

  timed_out_last_time = FALSE;
  return terminator;
}

int os_read_file_name (char *file_name, const char *default_name, int flag)
{
  char buf[INPUT_BUFFER_SIZE], prompt[INPUT_BUFFER_SIZE];
  FILE *fp;
  char *tempname;
  int i;

  /* If we're restoring a game before the interpreter starts,
   * our filename is already provided.  Just go ahead silently.
   */
  if (f_setup.restore_mode) {
    strcpy(file_name, default_name);
    return TRUE;
  } else {
    sprintf(prompt, "Please enter a filename [%s]: ", default_name);
    dumb_read_misc_line(buf, prompt);
    if (strlen(buf) > MAX_FILE_NAME) {
      printf("Filename too long\n");
      return FALSE;
    }
  }

  strcpy (file_name, buf[0] ? buf : default_name);

  /* Check if we're restricted to one directory. */
  if (f_setup.restricted_path != NULL) {
    for (i = strlen(file_name); i > 0; i--) {
      if (file_name[i] == PATH_SEPARATOR) {
        i++;
        break;
      }
    }
    tempname = strdup(file_name + i);
    strcpy(file_name, f_setup.restricted_path);
    if (file_name[strlen(file_name)-1] != PATH_SEPARATOR) {
      strcat(file_name, "/");
    }
    strcat(file_name, tempname);
  }

  /* Warn if overwriting a file.  */
  if ((flag == FILE_SAVE || flag == FILE_SAVE_AUX || flag == FILE_RECORD)
      && ((fp = fopen(file_name, "rb")) != NULL)) {
    fclose (fp);
    dumb_read_misc_line(buf, "Overwrite existing file? ");
    return(tolower(buf[0]) == 'y');
  }
  return TRUE;
}

void os_more_prompt (void)
{
  if (do_more_prompts) {
    char buf[INPUT_BUFFER_SIZE];
    dumb_read_misc_line(buf, "***MORE***");
  } else
    dumb_elide_more_prompt();
}

void dumb_init_input(void)
{
  if ((h_version >= V4) && (speed != 0))
    h_config |= CONFIG_TIMEDINPUT;

  if (h_version >= V5)
    h_flags &= ~(MOUSE_FLAG | MENU_FLAG);
}

zword os_read_mouse(void)
{
	/* NOT IMPLEMENTED */
    return 0;
}

void os_tick()
{}

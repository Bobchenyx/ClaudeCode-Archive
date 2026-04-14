#include "config.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Config* config_create(void) {
    Config* cfg = (Config*)calloc(1, sizeof(Config));
    if (!cfg) return NULL;
    cfg->count = 0;
    cfg->filepath[0] = '\0';
    return cfg;
}

void config_destroy(Config* cfg) {
    if (cfg) free(cfg);
}

static int find_entry(const Config* cfg, const char* key) {
    if (!cfg || !key) return -1;
    for (size_t i = 0; i < cfg->count; i++) {
        if (strcmp(cfg->entries[i].key, key) == 0) {
            return (int)i;
        }
    }
    return -1;
}

int config_load(Config* cfg, const char* filepath) {
    if (!cfg || !filepath) return -1;

    FILE* f = fopen(filepath, "r");
    if (!f) return -1;

    strncpy(cfg->filepath, filepath, sizeof(cfg->filepath) - 1);
    cfg->filepath[sizeof(cfg->filepath) - 1] = '\0';
    cfg->count = 0;

    char line[CONFIG_MAX_KEY_LEN + CONFIG_MAX_VAL_LEN + 2];
    while (fgets(line, sizeof(line), f)) {
        // Skip comments and empty lines
        char* trimmed = str_trim(line);
        if (trimmed[0] == '#' || trimmed[0] == '\0') continue;

        // Find '=' separator
        char* eq = strchr(trimmed, '=');
        if (!eq) continue;

        *eq = '\0';
        char* key = str_trim(trimmed);
        char* value = str_trim(eq + 1);

        if (cfg->count >= CONFIG_MAX_ENTRIES) {
            fclose(f);
            return -2; // Too many entries
        }

        strncpy(cfg->entries[cfg->count].key, key, CONFIG_MAX_KEY_LEN - 1);
        cfg->entries[cfg->count].key[CONFIG_MAX_KEY_LEN - 1] = '\0';
        strncpy(cfg->entries[cfg->count].value, value, CONFIG_MAX_VAL_LEN - 1);
        cfg->entries[cfg->count].value[CONFIG_MAX_VAL_LEN - 1] = '\0';
        cfg->count++;
    }

    fclose(f);
    return 0;
}

int config_save(const Config* cfg) {
    if (!cfg || cfg->filepath[0] == '\0') return -1;

    FILE* f = fopen(cfg->filepath, "w");
    if (!f) return -1;

    for (size_t i = 0; i < cfg->count; i++) {
        fprintf(f, "%s = %s\n", cfg->entries[i].key, cfg->entries[i].value);
    }

    fclose(f);
    return 0;
}

const char* config_get(const Config* cfg, const char* key) {
    int idx = find_entry(cfg, key);
    if (idx < 0) return NULL;
    return cfg->entries[idx].value;
}

int config_set(Config* cfg, const char* key, const char* value) {
    if (!cfg || !key || !value) return -1;

    int idx = find_entry(cfg, key);
    if (idx >= 0) {
        // Update existing
        strncpy(cfg->entries[idx].value, value, CONFIG_MAX_VAL_LEN - 1);
        cfg->entries[idx].value[CONFIG_MAX_VAL_LEN - 1] = '\0';
        return 0;
    }

    // Add new
    if (cfg->count >= CONFIG_MAX_ENTRIES) return -2;

    strncpy(cfg->entries[cfg->count].key, key, CONFIG_MAX_KEY_LEN - 1);
    cfg->entries[cfg->count].key[CONFIG_MAX_KEY_LEN - 1] = '\0';
    strncpy(cfg->entries[cfg->count].value, value, CONFIG_MAX_VAL_LEN - 1);
    cfg->entries[cfg->count].value[CONFIG_MAX_VAL_LEN - 1] = '\0';
    cfg->count++;
    return 0;
}

int config_remove(Config* cfg, const char* key) {
    if (!cfg || !key) return -1;

    int idx = find_entry(cfg, key);
    if (idx < 0) return -1;

    // Shift remaining entries
    for (size_t i = (size_t)idx; i < cfg->count - 1; i++) {
        cfg->entries[i] = cfg->entries[i + 1];
    }
    cfg->count--;
    return 0;
}

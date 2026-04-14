#ifndef CONFIG_H
#define CONFIG_H

#include <stddef.h>

#define CONFIG_MAX_KEY_LEN 64
#define CONFIG_MAX_VAL_LEN 256
#define CONFIG_MAX_ENTRIES 128

typedef struct {
    char key[CONFIG_MAX_KEY_LEN];
    char value[CONFIG_MAX_VAL_LEN];
} ConfigEntry;

typedef struct {
    ConfigEntry entries[CONFIG_MAX_ENTRIES];
    size_t count;
    char filepath[512];
} Config;

Config* config_create(void);
void config_destroy(Config* cfg);
int config_load(Config* cfg, const char* filepath);
int config_save(const Config* cfg);
const char* config_get(const Config* cfg, const char* key);
int config_set(Config* cfg, const char* key, const char* value);
int config_remove(Config* cfg, const char* key);

#endif // CONFIG_H

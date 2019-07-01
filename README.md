# virtual-skylight

```cron
*   5-7   * * *   cd virtual-skylight && systemd-cat -t virtual-skylight make run
*/5 8-9   * * *   cd virtual-skylight && systemd-cat -t virtual-skylight make run
# TODO: motion detection
1   9     * * 1-5 cd virtual-skylight && systemd-cat -t virtual-skylight make off
*   17-19 * * *   cd virtual-skylight && systemd-cat -t virtual-skylight make run
*/5 20    * * *   cd virtual-skylight && systemd-cat -t virtual-skylight make run
1   21    * * *   cd virtual-skylight && systemd-cat -t virtual-skylight make off
```

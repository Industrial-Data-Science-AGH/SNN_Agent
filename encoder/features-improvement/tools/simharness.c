/*
 * simharness.c — uruchamia PRAWDZIWY firmware (fw.elf, ATmega328P @16 MHz) w symulatorze simavr,
 * wstrzykuje próbki ADC (kody 10-bit) co SAMPLE_CYCLES cykli, mierzy liczbę cykli każdego wywołania
 * ISR(ADC_vect) i zbiera wydruk UART (linie debug enkodera).
 *
 * Użycie:  simharness fw.elf codes.bin [--hop 192] [--wait 200000] [--period 832] [--realtime]
 *   codes.bin : uint16 little-endian, jedna próbka ADC (0..1023) na wpis
 *   tryb domyślny ("wyrównany"): po każdych --hop próbkach wstrzykiwanie wstrzymane na --wait cykli,
 *     aż firmware przetworzy ramkę — ramki są WYRÓWNANE do siatki 192 próbek jak w twinie.
 *   --realtime : próbki płyną bez przerwy (jak na sprzęcie); loop() konkuruje z ISR o CPU.
 * stdout: to, co firmware wypisał na UART.   stderr: statystyki ISR (cykle, min/średnia/max).
 *
 * Model: ADC podmieniony — ADCL/ADCH zwracają bieżącą próbkę, zapis do ADCSRA tylko zapamiętywany
 * (bez wewn. konwersji simavr), przerwanie ADC (wektor 21) podnosimy sami.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "sim_avr.h"
#include "sim_elf.h"
#include "sim_core.h"
#include "avr_uart.h"
#include "sim_irq.h"

#define ADCL_ADDR   0x78
#define ADCH_ADDR   0x79
#define ADCSRA_ADDR 0x7A
#define ADC_VECTOR  21
#define ADC_VEC_PC  0x54          /* 21 * 4 bajty */

static uint16_t cur_sample = 512;

static uint8_t rd_l(avr_t *a, avr_io_addr_t ad, void *p) { (void)a; (void)ad; (void)p; return cur_sample & 0xFF; }
static uint8_t rd_h(avr_t *a, avr_io_addr_t ad, void *p) { (void)a; (void)ad; (void)p; return (cur_sample >> 8) & 0xFF; }
static void    wr_ctl(avr_t *a, avr_io_addr_t ad, uint8_t v, void *p) { (void)p; a->data[ad] = v; }
static void    uart_out(avr_irq_t *irq, uint32_t v, void *p) { (void)irq; (void)p; putchar(v & 0xFF); }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "użycie: %s fw.elf codes.bin [--hop N] [--wait C] [--period C] [--realtime]\n", argv[0]); return 2; }
    const char *send = ""; int hop = 192, realtime = 0; uint64_t wait_c = 200000, period = 832;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--hop") && i + 1 < argc) hop = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--wait") && i + 1 < argc) wait_c = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--period") && i + 1 < argc) period = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--realtime")) realtime = 1;
        else if (!strcmp(argv[i], "--send") && i + 1 < argc) send = argv[++i];
    }
    FILE *f = fopen(argv[2], "rb");
    if (!f) { perror("codes"); return 2; }
    fseek(f, 0, SEEK_END); long nb = ftell(f); fseek(f, 0, SEEK_SET);
    long ns = nb / 2; uint16_t *codes = malloc(nb);
    if (fread(codes, 2, ns, f) != (size_t)ns) { fprintf(stderr, "błąd odczytu\n"); return 2; }
    fclose(f);

    elf_firmware_t fw; memset(&fw, 0, sizeof fw);
    if (elf_read_firmware(argv[1], &fw)) { fprintf(stderr, "nie mogę wczytać ELF\n"); return 2; }
    strcpy(fw.mmcu, "atmega328p"); fw.frequency = 16000000;
    avr_t *avr = avr_make_mcu_by_name("atmega328p");
    if (!avr) { fprintf(stderr, "brak MCU\n"); return 2; }
    avr_init(avr);
    avr_load_firmware(avr, &fw);
    avr->frequency = 16000000;

    /* podmiana ADC: odczyty ADCL/ADCH z bieżącej próbki; zapis ADCSRA bez uruchamiania konwersji */
    avr->io[AVR_DATA_TO_IO(ADCL_ADDR)].r.c = rd_l;   avr->io[AVR_DATA_TO_IO(ADCL_ADDR)].r.param = NULL;
    avr->io[AVR_DATA_TO_IO(ADCH_ADDR)].r.c = rd_h;   avr->io[AVR_DATA_TO_IO(ADCH_ADDR)].r.param = NULL;
    avr->io[AVR_DATA_TO_IO(ADCSRA_ADDR)].w.c = wr_ctl; avr->io[AVR_DATA_TO_IO(ADCSRA_ADDR)].w.param = NULL;

    avr_irq_t *iq = avr_io_getirq(avr, AVR_IOCTL_UART_GETIRQ('0'), UART_IRQ_OUTPUT);
    avr_irq_register_notify(iq, uart_out, NULL);

    avr_irq_t *uart_in = avr_io_getirq(avr, AVR_IOCTL_UART_GETIRQ('0'), UART_IRQ_INPUT);
    avr_int_vector_t *adc_vec = NULL;               /* tablica vector[] jest w kolejności rejestracji, nie numeru */
    for (int i = 0; i < 64; i++) if (avr->interrupts.vector[i] && avr->interrupts.vector[i]->vector == ADC_VECTOR) adc_vec = avr->interrupts.vector[i];
    if (!adc_vec) { fprintf(stderr, "brak wektora ADC w symulatorze\n"); return 2; }

    uint64_t isr_n = 0, isr_sum = 0, isr_min = ~0ULL, isr_max = 0, isr_start = 0; int in_isr = 0;
    uint64_t next_sample = 20000000ULL;          /* 1.25 s na start firmware (setup, Serial, ...) */
    uint64_t t_first = 0; long si = 0; int in_frame = 0; uint64_t pause_until = 0; int paused = 0;

    /* rozgrzewka: dopuść setup() */
    while (avr->cycle < next_sample) { if (avr_run(avr) == cpu_Done || avr->state == cpu_Crashed) { fprintf(stderr, "crash w setup\n"); return 3; } }

    for (const char *c = send; *c; c++) avr_raise_irq(uart_in, (uint8_t)*c);   /* polecenia dla firmware (np. "B") — PO setup() */

    while (si < ns || (paused && avr->cycle < pause_until)) {
        if (paused) {
            if (avr->cycle >= pause_until) { paused = 0; next_sample = avr->cycle; }
        } else if (si < ns && avr->cycle >= next_sample) {
            if (si == 0) t_first = avr->cycle;
            cur_sample = codes[si++] & 0x3FF;
            avr_raise_interrupt(avr, adc_vec);
            next_sample += period;
            if (!realtime && ++in_frame >= hop) { in_frame = 0; paused = 1; pause_until = avr->cycle + wait_c + 4000; }
        }
        uint32_t pc_before = avr->pc;
        int is_reti = (avr->flash[pc_before] == 0x18 && avr->flash[pc_before + 1] == 0x95);
        avr_run(avr);
        if (avr->state == cpu_Crashed || avr->state == cpu_Done) { fprintf(stderr, "crash/done @ pc=0x%x\n", avr->pc); return 3; }
        if (!in_isr && avr->pc == ADC_VEC_PC && pc_before != ADC_VEC_PC) { in_isr = 1; isr_start = avr->cycle - 4; }
        else if (in_isr && is_reti) {
            uint64_t c = avr->cycle - isr_start; in_isr = 0;
            isr_n++; isr_sum += c; if (c < isr_min) isr_min = c; if (c > isr_max) isr_max = c;
        }
    }
    /* domknięcie ostatniej ramki */
    uint64_t end = avr->cycle + wait_c;
    while (avr->cycle < end) avr_run(avr);
    fflush(stdout);
    fprintf(stderr, "ISR: n=%llu min=%llu mean=%.1f max=%llu cykli (od wektora, z wejściem)\n",
            (unsigned long long)isr_n, (unsigned long long)isr_min, isr_n ? (double)isr_sum / isr_n : 0.0, (unsigned long long)isr_max);
    fprintf(stderr, "INJECT: wstrzyknięto=%ld obsłużono=%llu (zgubione=%ld)  zajętość CPU przez ISR=%.1f%%\n", si, (unsigned long long)isr_n,
            si - (long)isr_n, 100.0 * (double)isr_sum / (double)((avr->cycle - t_first) ? (avr->cycle - t_first) : 1));
    return 0;
}
